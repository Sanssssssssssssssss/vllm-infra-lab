#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:?workspace path is required}"
shift
VENV_DIR="${HOME}/.venvs/gptproject2-vllm"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/bootstrap_vllm_wsl.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"
python "${WORKSPACE_DIR}/scripts/test_openai_api.py" "$@"

