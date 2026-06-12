#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

CONDA_BIN="${CONDA_BIN:-conda}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda first, or set CONDA_BIN=/path/to/conda." >&2
  exit 1
fi

cd "${WORK_DIR}"

create_or_update_env() {
  local env_name="$1"
  local env_file="$2"

  if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "Updating conda env: ${env_name}"
    "${CONDA_BIN}" env update -n "${env_name}" -f "${env_file}" --prune
  else
    echo "Creating conda env: ${env_name}"
    "${CONDA_BIN}" env create -f "${env_file}"
  fi
}

create_or_update_env "searchr1" "${SCRIPT_DIR}/environment-searchr1.yml"
create_or_update_env "retriever" "${SCRIPT_DIR}/environment-retriever.yml"

echo "Installing Search-R1 runtime packages into searchr1"
"${CONDA_BIN}" run -n searchr1 pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
"${CONDA_BIN}" run -n searchr1 pip install vllm==0.6.3
"${CONDA_BIN}" run -n searchr1 pip install -e "${WORK_DIR}"
"${CONDA_BIN}" run -n searchr1 pip install flash-attn --no-build-isolation
"${CONDA_BIN}" run -n searchr1 pip install wandb huggingface_hub

echo "Environment setup complete."
echo "Use: conda activate searchr1    # training/evaluation"
echo "Use: conda activate retriever   # retrieval server"
