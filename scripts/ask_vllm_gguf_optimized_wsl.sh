#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:?workspace path is required}"
MESSAGE="${2:?message is required}"
MAX_TOKENS="${3:-64}"
TIMEOUT="${4:-180}"
THINKING_MODE="${5:-off}"

VENV_DIR="${VENV_DIR:-${HOME}/.venvs/gptproject2-vllm}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/bootstrap_vllm_wsl.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"

args=(
  "${WORKSPACE_DIR}/scripts/test_openai_api.py"
  --host "${VLLM_HOST:-127.0.0.1}"
  --port "${VLLM_PORT:-8000}"
  --api-key "${VLLM_API_KEY:-change-this-before-lan-use}"
  --model "${VLLM_SERVED_MODEL_NAME:-Qwen3-8B-GGUF-vLLM-local}"
  --message "${MESSAGE}"
  --max-tokens "${MAX_TOKENS}"
  --timeout "${TIMEOUT}"
)

if [ "${THINKING_MODE}" != "on" ]; then
  args+=(--disable-thinking)
fi

python "${args[@]}"
