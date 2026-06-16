#!/usr/bin/env bash
set -euo pipefail

# Rebuild the current Python environment for a vLLM 0.5.x compatibility image.
# Run this inside the xFusion online-development terminal, then test vLLM before
# using "Make Image" in the platform UI.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-${PYTHON_BIN} -m pip}"
VLLM_VERSION="${VLLM_VERSION:-0.5.3.post1}"
TORCH_VERSION="${TORCH_VERSION:-2.3.0}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers<4.48}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"

echo "== Python =="
"${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY

echo "== Before =="
"${PYTHON_BIN}" - <<'PY' || true
for name in ["torch", "vllm", "transformers", "flash_attn"]:
    try:
        mod = __import__(name)
        print(f"{name}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{name}: missing ({exc})")
PY

echo "== Installing packaging tools =="
${PIP_BIN} install --no-cache-dir --upgrade pip setuptools wheel packaging

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  echo "== Installing torch ${TORCH_VERSION}+cu121 for vLLM ${VLLM_VERSION} =="
  ${PIP_BIN} uninstall -y torch torchvision torchaudio xformers || true
  ${PIP_BIN} install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu121
fi

echo "== Installing vLLM ${VLLM_VERSION} without allowing it to replace torch =="
${PIP_BIN} uninstall -y vllm || true
${PIP_BIN} install --no-cache-dir --no-deps "vllm==${VLLM_VERSION}"

echo "== Pinning compatible high-level packages =="
${PIP_BIN} install --no-cache-dir "${TRANSFORMERS_SPEC}" datasets pyserini uvicorn fastapi huggingface_hub wandb IPython matplotlib pyairports

echo "== Repairing pyairports import if the wheel only installed metadata =="
"${PYTHON_BIN}" - <<'PY'
import pathlib
import site
import sys

try:
    import pyairports  # noqa: F401
except Exception:
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    candidates.append(next((p for p in sys.path if p.endswith("dist-packages")), sys.path[-1]))
    target_root = pathlib.Path(candidates[0])
    pkg = target_root / "pyairports"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "airports.py").write_text("AIRPORT_LIST = []\n", encoding="utf-8")
    print(f"created pyairports import shim at {pkg}")
else:
    print("pyairports import OK")
PY

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "== Reinstalling flash-attn against the current torch/CUDA =="
  ${PIP_BIN} uninstall -y flash-attn || true
  ${PIP_BIN} install --no-cache-dir flash-attn --no-build-isolation
fi

echo "== Installing Search-R1 editable package =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
${PIP_BIN} install --no-cache-dir -e "${WORK_DIR}"

echo "== After =="
"${PYTHON_BIN}" - <<'PY'
import sys
print("python:", sys.executable)
for name in ["torch", "vllm", "transformers"]:
    mod = __import__(name)
    print(f"{name}: {getattr(mod, '__version__', 'ok')}")
try:
    import flash_attn
    print("flash_attn: ok")
except Exception as exc:
    print("flash_attn:", exc)
try:
    import torch
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as exc:
    print("torch cuda check failed:", exc)
PY

echo "vLLM ${VLLM_VERSION} install complete."
echo "Next: python3 reproduction/7b_base/test_vllm_llama.py"
