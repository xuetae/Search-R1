#!/usr/bin/env python3
"""Minimal vLLM Llama generation check for the xFusion H20 image."""

from __future__ import annotations

import os

from vllm import LLM, SamplingParams


def main() -> None:
    model = os.environ.get("LOCAL_BASE_MODEL", "/workspace/filesdir/models/7b_base/llama-7b")
    dtype = os.environ.get("VLLM_TEST_DTYPE", "float16")
    max_model_len = int(os.environ.get("VLLM_TEST_MAX_MODEL_LEN", "1024"))
    max_num_batched_tokens = int(os.environ.get("VLLM_TEST_MAX_NUM_BATCHED_TOKENS", str(max_model_len)))
    max_num_seqs = int(os.environ.get("VLLM_TEST_MAX_NUM_SEQS", "1"))
    gpu_memory_utilization = float(os.environ.get("VLLM_TEST_GPU_MEMORY_UTILIZATION", "0.2"))
    enforce_eager = os.environ.get("VLLM_TEST_ENFORCE_EAGER", "true").lower() in {"1", "true", "yes"}

    print("model:", model, flush=True)
    print("dtype:", dtype, flush=True)
    print("enforce_eager:", enforce_eager, flush=True)
    print("max_model_len:", max_model_len, flush=True)
    print("max_num_batched_tokens:", max_num_batched_tokens, flush=True)
    print("max_num_seqs:", max_num_seqs, flush=True)

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    outputs = llm.generate(
        ["Question: Who is the president of the United States?\nAnswer:"],
        SamplingParams(max_tokens=32, temperature=1.0),
    )
    print(outputs[0].outputs[0].text, flush=True)
    print("vllm llama generation OK", flush=True)


if __name__ == "__main__":
    main()
