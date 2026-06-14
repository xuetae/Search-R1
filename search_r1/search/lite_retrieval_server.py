import argparse
import json
import re
from collections import Counter
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def load_lite_corpus(corpus_path: str, max_docs: int):
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_docs:
                break
            item = json.loads(line)
            contents = item.get("contents") or item.get("text") or ""
            if not contents:
                continue
            title = item.get("title")
            if title is None:
                title = contents.split("\n", 1)[0].strip('"')
            docs.append(
                {
                    "id": item.get("id", str(idx)),
                    "title": title,
                    "text": item.get("text", "\n".join(contents.split("\n")[1:])),
                    "contents": contents,
                    "_tf": Counter(tokenize(contents)),
                }
            )
    return docs


class LiteRetriever:
    def __init__(self, corpus_path: str, max_docs: int):
        self.docs = load_lite_corpus(corpus_path, max_docs=max_docs)
        if not self.docs:
            raise RuntimeError(f"No documents loaded from {corpus_path}")

    def search(self, query: str, topk: int):
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_counts = Counter(q_tokens)
        scored = []
        for doc in self.docs:
            score = 0.0
            tf = doc["_tf"]
            for token, count in q_counts.items():
                if token in tf:
                    score += min(count, tf[token])
            if score > 0:
                scored.append((score, doc))

        if not scored:
            # Return deterministic fallback docs so the training path still gets observations.
            scored = [(0.0, doc) for doc in self.docs[:topk]]

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:topk]:
            result_doc = {k: v for k, v in doc.items() if k != "_tf"}
            results.append({"document": result_doc, "score": float(score)})
        return results


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever = None
default_topk = 1


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or default_topk
    results = []
    for query in request.queries:
        hits = retriever.search(query, topk=topk)
        if request.return_scores:
            results.append(hits)
        else:
            results.append([hit["document"] for hit in hits])
    return {"result": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a memory-light retrieval server for smoke tests.")
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--max_docs", type=int, default=20000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    default_topk = args.topk
    retriever = LiteRetriever(corpus_path=args.corpus_path, max_docs=args.max_docs)
    print(f"Loaded {len(retriever.docs)} lite retrieval docs from {args.corpus_path}")
    uvicorn.run(app, host=args.host, port=args.port)
