#!/usr/bin/env python3
"""Build small SFT datasets for Search-R1-lite.

The script derives query, evidence-QA, decision, or mixed SFT data from the
already prepared NQ Search-R1 parquet files. Evidence tasks optionally call the
local retriever and keep only examples whose retrieved text contains a golden
answer, which keeps the SFT signal clean for small models.
"""

import argparse
import os
import re
from typing import Iterable

import pandas as pd
import requests


QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n|$)", re.DOTALL)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def extract_question(prompt) -> str:
    if isinstance(prompt, list):
        content = " ".join(str(item.get("content", "")) for item in prompt)
    else:
        content = str(prompt)
    match = QUESTION_RE.search(content)
    if match:
        return " ".join(match.group(1).strip().split())
    return " ".join(content.strip().split())


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


def retrieve(question: str, retriever_url: str, topk: int, timeout: int) -> str:
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
        text = " ".join(text.split())
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
        if args.task in {"evidence_qa", "decision", "mixed"}:
            try:
                evidence = retrieve(question, args.retriever_url, args.topk, args.timeout)
            except Exception as exc:
                print(f"[WARN] retrieval failed at {split}:{idx}: {exc}")
                continue
            if args.filter_answer_in_evidence and not answer_in_evidence(evidence, answers):
                continue

        if args.task in {"query", "mixed"}:
            yield build_query_row(question, answers, split, idx)
        if args.task in {"evidence_qa", "mixed"} and evidence:
            yield build_evidence_qa_row(question, evidence, answers, split, idx)
        if args.task in {"decision", "mixed"}:
            yield from build_decision_rows(question, evidence, answers, split, idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/nq_search")
    parser.add_argument("--local_dir", default="data/nq_search_lite_sft")
    parser.add_argument("--task", choices=["query", "evidence_qa", "decision", "mixed"], default="mixed")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
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
