#!/usr/bin/env bash
set -euo pipefail

# Restore the current xFusion H20 Python environment to the most compatible
# Hugging Face / PyTorch inference setup for Llama-7B. This intentionally
# removes vLLM/xFormers/flash-attn because vLLM generation SIGFPEs on the
# current H20 image.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-${PYTHON_BIN} -m pip}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.47.1}"

echo "== Python =="
"${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY

echo "== Removing unstable / conflicting GPU inference packages =="
${PIP_BIN} uninstall -y vllm vllm-flash-attn xformers flash-attn torchvision torchaudio || true

echo "== Installing PyTorch ${TORCH_VERSION}+cu121 =="
${PIP_BIN} install --no-cache-dir --upgrade pip setuptools wheel packaging
${PIP_BIN} install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu121

echo "== Installing compatible HF inference stack =="
${PIP_BIN} install --no-cache-dir \
  "${TRANSFORMERS_SPEC}" \
  accelerate \
  safetensors \
  sentencepiece \
  protobuf \
  huggingface_hub \
  datasets \
  uvicorn \
  fastapi \
  wandb \
  IPython \
  matplotlib

echo "== Installing Search-R1 editable package =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
${PIP_BIN} install --no-cache-dir -e "${WORK_DIR}"

echo "== After =="
"${PYTHON_BIN}" - <<'PY'
import sys
import torch
import transformers
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"gpu {idx}:", torch.cuda.get_device_name(idx))
print("transformers:", transformers.__version__)
try:
    import vllm
    print("vllm: still installed", getattr(vllm, "__version__", "unknown"))
except Exception as exc:
    print("vllm: unavailable as expected:", exc.__class__.__name__)
PY

echo "HF H20 environment restore complete."
echo "Next: CUDA_VISIBLE_DEVICES=0,1 python3 reproduction/7b_base/test_hf_llama_inference.py"
