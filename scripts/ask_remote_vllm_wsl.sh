#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?host is required}"
API_KEY="${2:?api key is required}"
MESSAGE="${3:?message is required}"
MODEL_NAME="${4:-Qwen3-8B-local}"
MAX_TOKENS="${5:-64}"
TIMEOUT="${6:-120}"
THINKING_MODE="${7:-off}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=(
  python3 "${SCRIPT_DIR}/test_openai_api.py"
  --host "${HOST}"
  --port 8000
  --api-key "${API_KEY}"
  --model "${MODEL_NAME}"
  --message "${MESSAGE}"
  --max-tokens "${MAX_TOKENS}"
  --timeout "${TIMEOUT}"
)

if [ "${THINKING_MODE}" != "on" ]; then
  args+=(--disable-thinking)
fi

"${args[@]}"
