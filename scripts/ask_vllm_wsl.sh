#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:?workspace path is required}"
MESSAGE="${2:?message is required}"
MAX_TOKENS="${3:-64}"
TIMEOUT="${4:-120}"
THINKING_MODE="${5:-off}"

VENV_DIR="${HOME}/.venvs/gptproject2-vllm"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/bootstrap_vllm_wsl.sh first."
  exit 1
fi

cd "${WORKSPACE_DIR}"
source config/runtime.env
source "${VENV_DIR}/bin/activate"

args=(
  "${WORKSPACE_DIR}/scripts/test_openai_api.py"
  --host 127.0.0.1
  --port "${VLLM_PORT}"
  --api-key "${VLLM_API_KEY}"
  --model "${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
  --message "${MESSAGE}"
  --max-tokens "${MAX_TOKENS}"
  --timeout "${TIMEOUT}"
)

if [ "${THINKING_MODE}" != "on" ]; then
  args+=(--disable-thinking)
fi

python "${args[@]}"
