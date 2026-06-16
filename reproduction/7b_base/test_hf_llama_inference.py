#!/usr/bin/env python3
"""Two-GPU compatible Hugging Face Llama inference check for H20."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_path = os.environ.get("LOCAL_BASE_MODEL", "/workspace/filesdir/models/7b_base/llama-7b")
    attention = os.environ.get("HF_ATTENTION", "sdpa")
    max_new_tokens = int(os.environ.get("HF_MAX_NEW_TOKENS", "64"))
    do_sample = os.environ.get("HF_DO_SAMPLE", "true").lower() in {"1", "true", "yes"}

    print("model:", model_path, flush=True)
    print("attention:", attention, flush=True)
    print("cuda visible:", os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)
    print("gpu count:", torch.cuda.device_count(), flush=True)
    for idx in range(torch.cuda.device_count()):
        print(f"gpu {idx}: {torch.cuda.get_device_name(idx)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = {idx: os.environ.get("HF_GPU_MAX_MEMORY", "70GiB") for idx in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory or None,
        attn_implementation=attention,
    )

    prompt = os.environ.get("HF_TEST_PROMPT", "Question: Who is the president of the United States?\nAnswer:")
    inputs = tokenizer(prompt, return_tensors="pt")
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=float(os.environ.get("HF_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("HF_TOP_P", "0.9")),
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True), flush=True)
    print("HF transformers inference OK", flush=True)


if __name__ == "__main__":
    main()
