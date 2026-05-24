#!/usr/bin/env python3
"""Build small SFT datasets for Search-R1-lite.

The script derives query, evidence-QA, decision, or mixed SFT data from the
already prepared NQ Search-R1 parquet files. Evidence tasks optionally call the
local retriever and keep only examples whose retrieved text contains a golden
answer, which keeps the SFT signal clean for small models.
"""

import argparse
import ast
import os
import re
from typing import Iterable

import pandas as pd
import requests


QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n|$)", re.DOTALL)
SEARCH_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new "
    "information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> "
    "query </search> and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: {question}"
)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def extract_question(prompt) -> str:
    content = ""
    if isinstance(prompt, list):
        content = " ".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in prompt)
    elif hasattr(prompt, "tolist"):
        prompt_list = prompt.tolist()
        if isinstance(prompt_list, list):
            content = " ".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in prompt_list)
    if not content:
        content = str(prompt)
        if content.startswith("[") and "content" in content:
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    content = " ".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in parsed)
            except (SyntaxError, ValueError):
                pass
    match = QUESTION_RE.search(content)
    if match:
        question = match.group(1)
    else:
        question = content
    question = question.replace("\\n", "\n").splitlines()[0]
    question = re.sub(r"""['"}\]\)]+$""", "", question.strip())
    return " ".join(question.split())


def extract_answers(reward_model) -> list[str]:
    ground_truth = reward_model.get("ground_truth", reward_model)
    target = ground_truth.get("target", [])
    if isinstance(target, str):
        return [target]
    return [str(item) for item in target]


def concise_query(question: str) -> str:
    query = question.strip().rstrip("?")
    query = re.sub(r"^(who|what|when|where|which|why|how)\s+(is|are|was|were|does|do|did|has|have|had|the)\s+", "", query, flags=re.I)
    query = re.sub(r"\b(the|a|an|of|in|on|at|to|for|by|with|from|does|do|did|is|are|was|were)\b", " ", query, flags=re.I)
    query = " ".join(query.split())
    return query or question.strip().rstrip("?")


def prompt(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def sentence_score(question: str, sentence: str) -> int:
    q_terms = set(normalize_answer(question).split())
    s_terms = set(normalize_answer(sentence).split())
    if not q_terms or not s_terms:
        return 0
    return len(q_terms & s_terms)


def compress_text(question: str, text: str, max_sentences: int, max_chars: int) -> str:
    sentences = [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if sent.strip()]
    if not sentences:
        return " ".join(text.split())[:max_chars]
    ranked = sorted(sentences, key=lambda sent: sentence_score(question, sent), reverse=True)
    return " ".join(ranked[:max_sentences])[:max_chars].strip()


def retrieve(question: str, retriever_url: str, topk: int, timeout: int, max_sentences_per_doc: int, max_chars_per_doc: int) -> str:
    response = requests.post(
        retriever_url,
        json={"queries": [concise_query(question)], "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    docs = response.json()["result"][0]
    blocks = []
    for idx, doc in enumerate(docs):
        contents = doc["document"]["contents"]
        title, _, text = contents.partition("\n")
        text = compress_text(question, text, max_sentences_per_doc, max_chars_per_doc)
        blocks.append(f"Doc {idx + 1}(Title: {title}) {text}")
    return "\n".join(blocks)


def answer_in_evidence(evidence: str, answers: Iterable[str]) -> bool:
    evidence_norm = normalize_answer(evidence)
    return any(normalize_answer(answer) in evidence_norm for answer in answers)


def build_query_row(question: str, answers: list[str], split: str, idx: int) -> dict:
    return {
        "prompt": prompt(f"Rewrite the question as a short keyword search query.\nQuestion: {question}"),
        "response": f"<think> I need to search for the key fact. </think>\n<search> {concise_query(question)} </search>",
        "task": "query",
        "split": split,
        "source_index": idx,
        "target": answers,
    }


def build_evidence_qa_row(question: str, evidence: str, answers: list[str], split: str, idx: int) -> dict:
    return {
        "prompt": prompt(
            "Answer the question using only the evidence. Return only the final answer in <answer> tags.\n"
            f"Question: {question}\nEvidence:\n{evidence}"
        ),
        "response": f"<think> The evidence contains the answer. </think>\n<answer> {answers[0]} </answer>",
        "task": "evidence_qa",
        "split": split,
        "source_index": idx,
        "target": answers,
    }


def build_decision_rows(question: str, evidence: str | None, answers: list[str], split: str, idx: int) -> list[dict]:
    rows = [{
        "prompt": prompt(
            "Decide the next action. If evidence is missing, search with a concise query. "
            "If evidence is sufficient, answer in <answer> tags.\n"
            f"Question: {question}"
        ),
        "response": f"<think> I need external evidence before answering. </think>\n<search> {concise_query(question)} </search>",
        "task": "decision_search",
        "split": split,
        "source_index": idx,
        "target": answers,
    }]
    if evidence:
        rows.append({
            "prompt": prompt(
                "Decide the next action. If evidence is missing, search with a concise query. "
                "If evidence is sufficient, answer in <answer> tags.\n"
                f"Question: {question}\nEvidence:\n{evidence}"
            ),
            "response": f"<think> The evidence is sufficient to answer. </think>\n<answer> {answers[0]} </answer>",
            "task": "decision_answer",
            "split": split,
            "source_index": idx,
            "target": answers,
        })
    return rows


def build_post_search_answer_row(question: str, evidence: str, answers: list[str], split: str, idx: int) -> dict:
    query = concise_query(question)
    content = (
        SEARCH_PROMPT.format(question=question)
        + f"\n\nPrevious search action:\n<think> I need external evidence. </think>\n<search> {query} </search>"
        + f"\n\n<information>{evidence}</information>\n\nUse the information to identify the supporting fact and answer."
    )
    return {
        "prompt": prompt(content),
        "response": f"<think> The relevant evidence supports the answer: {answers[0]}. </think>\n<answer> {answers[0]} </answer>",
        "task": "post_search_answer",
        "split": split,
        "source_index": idx,
        "target": answers,
    }


def iter_rows(df: pd.DataFrame, split: str, args):
    limit = args.train_limit if split == "train" else args.test_limit
    if limit > 0:
        df = df.head(limit)

    for idx, row in df.iterrows():
        question = extract_question(row["prompt"])
        answers = extract_answers(row["reward_model"])
        if not answers:
            continue

        evidence = None
        if args.task in {"evidence_qa", "decision", "post_search_answer", "grounding", "mixed"}:
            try:
                evidence = retrieve(question, args.retriever_url, args.topk, args.timeout,
                                    args.max_sentences_per_doc, args.max_chars_per_doc)
            except Exception as exc:
                print(f"[WARN] retrieval failed at {split}:{idx}: {exc}")
                continue
            if args.filter_answer_in_evidence and not answer_in_evidence(evidence, answers):
                continue

        if args.task in {"query", "mixed"}:
            yield build_query_row(question, answers, split, idx)
        if args.task in {"evidence_qa", "mixed"} and evidence:
            yield build_evidence_qa_row(question, evidence, answers, split, idx)
        if args.task in {"decision", "grounding", "mixed"}:
            yield from build_decision_rows(question, evidence, answers, split, idx)
        if args.task in {"post_search_answer", "grounding", "mixed"} and evidence:
            yield build_post_search_answer_row(question, evidence, answers, split, idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/nq_search")
    parser.add_argument("--local_dir", default="data/nq_search_lite_sft")
    parser.add_argument("--task", choices=["query", "evidence_qa", "decision", "post_search_answer", "grounding", "mixed"], default="mixed")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_sentences_per_doc", type=int, default=2)
    parser.add_argument("--max_chars_per_doc", type=int, default=360)
    parser.add_argument("--train_limit", type=int, default=5000)
    parser.add_argument("--test_limit", type=int, default=512)
    parser.add_argument("--filter_answer_in_evidence", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    for split in ["train", "test"]:
        path = os.path.join(args.input_dir, f"{split}.parquet")
        df = pd.read_parquet(path)
        rows = list(iter_rows(df, split, args))
        out_path = os.path.join(args.local_dir, f"{split}.parquet")
        pd.DataFrame(rows).to_parquet(out_path)
        print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
