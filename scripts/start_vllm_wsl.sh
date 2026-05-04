#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:?workspace path is required}"
VENV_DIR="${HOME}/.venvs/gptproject2-vllm"

cd "${WORKSPACE_DIR}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/bootstrap_vllm_wsl.ps1 first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"
source config/runtime.env

cmd=(
  vllm serve "$VLLM_MODEL"
  --served-model-name "${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --api-key "$VLLM_API_KEY"
  --dtype "$VLLM_DTYPE"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB"
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
)

if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
  cmd+=(--enforce-eager)
fi

exec "${cmd[@]}"
