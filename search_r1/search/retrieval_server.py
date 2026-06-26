import json
import os
import time
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

def load_corpus(corpus_path: str):
    datasets.disable_progress_bars()
    cache_dir = os.environ.get("RETRIEVER_DATASETS_CACHE") or os.environ.get("HF_DATASETS_CACHE")
    num_proc = int(os.environ.get("RETRIEVER_CORPUS_NUM_PROC", "1"))
    print(f"[retriever] loading corpus={corpus_path} cache_dir={cache_dir or '<default>'} num_proc={num_proc}", flush=True)
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    print(f"[retriever] loaded corpus rows={len(corpus)}", flush=True)
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False, device: str = "cuda"):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16 and device.startswith("cuda"):
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, device):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        self.model, self.tokenizer = load_model(
            model_path=model_path,
            use_fp16=self.use_fp16,
            device=str(self.device),
        )
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            temp_memory_mb = getattr(config, "faiss_gpu_temp_memory_mb", None)
            if temp_memory_mb is not None and temp_memory_mb >= 0:
                try:
                    resources = []
                    for _ in range(faiss.get_num_gpus()):
                        res = faiss.StandardGpuResources()
                        res.setTempMemory(int(temp_memory_mb) * 1024 * 1024)
                        resources.append(res)
                    self.index = faiss.index_cpu_to_gpu_multiple_py(resources, self.index, co=co)
                    self._faiss_gpu_resources = resources
                    print(
                        f"[retriever] faiss_gpu_temp_memory_mb={temp_memory_mb} "
                        f"gpus={len(resources)} useFloat16={co.useFloat16} shard={co.shard}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        "[retriever] failed to apply FAISS GPU temp memory limit; "
                        f"falling back to default resources: {exc}",
                        flush=True,
                    )
                    self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            else:
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16,
            device = config.retrieval_device,
            )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        encode_seconds = 0.0
        search_seconds = 0.0
        load_seconds = 0.0
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            started_at = time.perf_counter()
            batch_emb = self.encoder.encode(query_batch)
            encode_seconds += time.perf_counter() - started_at
            started_at = time.perf_counter()
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            search_seconds += time.perf_counter() - started_at
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            started_at = time.perf_counter()
            batch_results = load_docs(self.corpus, flat_idxs)
            load_seconds += time.perf_counter() - started_at
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results

        print(
            f"[retriever-breakdown] queries={len(query_list)} "
            f"encode_s={encode_seconds:.3f} faiss_s={search_seconds:.3f} "
            f"load_docs_s={load_seconds:.3f}",
            flush=True,
        )
            
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        retrieval_device: str = "cuda",
        retrieval_max_return_tokens: int = 400,
        faiss_gpu_temp_memory_mb: int | None = None,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_device = retrieval_device
        self.retrieval_max_return_tokens = retrieval_max_return_tokens
        self.faiss_gpu_temp_memory_mb = faiss_gpu_temp_memory_mb


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

def truncate_documents(documents, tokenizer, max_total_tokens):
    """Bound the total returned passage payload before HTTP serialization."""
    if max_total_tokens <= 0 or not documents:
        return documents

    per_document_budget = max(1, max_total_tokens // len(documents))
    truncated = []
    for document in documents:
        document = dict(document)
        contents = document.get("contents", "")
        contents_for_tokenize = contents[:per_document_budget * 8]
        token_ids = tokenizer.encode(contents_for_tokenize, add_special_tokens=False)
        if len(token_ids) > per_document_budget:
            contents = tokenizer.decode(
                token_ids[:per_document_budget],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            document["contents"] = contents
            lines = contents.split("\n", 1)
            document["title"] = lines[0].strip('"')
            document["text"] = lines[1] if len(lines) > 1 else ""
        truncated.append(document)
    return truncated

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    started_at = time.perf_counter()
    search_output = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    if request.return_scores:
        results, scores = search_output
    else:
        results, scores = search_output, None
    tokenizer = getattr(getattr(retriever, "encoder", None), "tokenizer", None)
    if tokenizer is not None:
        results = [
            truncate_documents(single_result, tokenizer, config.retrieval_max_return_tokens)
            for single_result in results
        ]
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    elapsed = time.perf_counter() - started_at
    print(
        f"[retriever] queries={len(request.queries)} topk={request.topk} "
        f"return_token_budget={config.retrieval_max_return_tokens} elapsed_s={elapsed:.3f}",
        flush=True,
    )
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--retriever_device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for query encoding.")
    parser.add_argument("--max_return_tokens", type=int, default=400, help="Maximum total returned passage tokens per query.")
    parser.add_argument(
        "--faiss_gpu_temp_memory_mb",
        type=int,
        default=None,
        help="Per-GPU FAISS scratch memory cap in MB. Keeps exact Flat GPU search but reduces resident temp memory.",
    )
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=args.retriever_device == "cuda",
        retrieval_batch_size=512,
        retrieval_device=args.retriever_device,
        retrieval_max_return_tokens=args.max_return_tokens,
        faiss_gpu_temp_memory_mb=args.faiss_gpu_temp_memory_mb,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
