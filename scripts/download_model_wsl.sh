#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/mnt/e/GPTProject2/vLLM/models/Qwen3-8B}"
MODEL_ID="${2:-Qwen/Qwen3-8B}"
VENV_DIR="${HOME}/.venvs/gptproject2-vllm"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/bootstrap_vllm_wsl.sh first."
  exit 1
fi

mkdir -p "${MODEL_DIR}"
source "${VENV_DIR}/bin/activate"
export HF_HOME="${HF_HOME:-/mnt/e/GPTProject2/vLLM/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
