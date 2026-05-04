#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:?workspace path is required}"
VENV_DIR="${HOME}/.venvs/gptproject2-vllm"

cd "${WORKSPACE_DIR}"

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "python3-venv is missing. Install it first inside Ubuntu:"
  echo "  sudo apt-get update && sudo apt-get install -y python3-venv python3-pip"
  exit 1
fi

mkdir -p "${HOME}/.venvs"
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install vllm==0.17.1 openai

