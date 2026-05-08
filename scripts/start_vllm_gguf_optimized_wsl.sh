#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/mnt/e/GPTProject2/vLLM}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/gptproject2-vllm}"

cd "${WORKSPACE_DIR}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/bootstrap_vllm_wsl.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"

export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MODEL_ROOT="${VLLM_MODEL_ROOT:-${WORKSPACE_DIR}/models}"

VLLM_MODEL="${VLLM_MODEL:-${MODEL_ROOT}/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf}"
VLLM_TOKENIZER="${VLLM_TOKENIZER:-${MODEL_ROOT}/Qwen3-8B}"
VLLM_HF_CONFIG_PATH="${VLLM_HF_CONFIG_PATH:-${MODEL_ROOT}/Qwen3-8B}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-Qwen3-8B-GGUF-vLLM-local}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-change-this-before-lan-use}"
VLLM_DTYPE="${VLLM_DTYPE:-half}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-vllm}"

cmd=(
  vllm serve "${VLLM_MODEL}"
  --tokenizer "${VLLM_TOKENIZER}"
  --hf-config-path "${VLLM_HF_CONFIG_PATH}"
  --served-model-name "${VLLM_SERVED_MODEL_NAME}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --api-key "${VLLM_API_KEY}"
  --dtype "${VLLM_DTYPE}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --block-size "${VLLM_BLOCK_SIZE}"
  --generation-config "${VLLM_GENERATION_CONFIG}"
)

if [ -n "${VLLM_QUANTIZATION:-}" ]; then
  cmd+=(--quantization "${VLLM_QUANTIZATION}")
fi

if [ -n "${VLLM_CHAT_TEMPLATE:-}" ]; then
  cmd+=(--chat-template "${VLLM_CHAT_TEMPLATE}")
fi

if [ "${VLLM_TRUST_REMOTE_CODE:-0}" = "1" ]; then
  cmd+=(--trust-remote-code)
fi

if [ "${VLLM_ENABLE_PREFIX_CACHING:-1}" = "1" ]; then
  cmd+=(--enable-prefix-caching)
fi

if [ -n "${VLLM_PREFIX_CACHING_HASH_ALGO:-}" ]; then
  cmd+=(--prefix-caching-hash-algo "${VLLM_PREFIX_CACHING_HASH_ALGO}")
fi

if [ "${VLLM_ENABLE_CHUNKED_PREFILL:-1}" = "1" ]; then
  cmd+=(--enable-chunked-prefill)
else
  cmd+=(--no-enable-chunked-prefill)
fi

if [ "${VLLM_ASYNC_SCHEDULING:-1}" = "1" ]; then
  cmd+=(--async-scheduling)
fi

if [ -n "${VLLM_KV_CACHE_DTYPE:-}" ]; then
  cmd+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
fi

if [ -n "${VLLM_KV_CACHE_MEMORY_BYTES:-}" ]; then
  cmd+=(--kv-cache-memory-bytes "${VLLM_KV_CACHE_MEMORY_BYTES}")
fi

if [ -n "${VLLM_KV_OFFLOADING_SIZE:-}" ]; then
  cmd+=(--kv-offloading-size "${VLLM_KV_OFFLOADING_SIZE}")
fi

if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
  cmd+=(--enforce-eager)
fi

if [ "${VLLM_ENABLE_LOGGING_ITERATION_DETAILS:-0}" = "1" ]; then
  cmd+=(--enable-logging-iteration-details)
fi

if [ "${VLLM_KV_CACHE_METRICS:-0}" = "1" ]; then
  cmd+=(--kv-cache-metrics)
fi

if [ "${VLLM_CUDAGRAPH_METRICS:-0}" = "1" ]; then
  cmd+=(--cudagraph-metrics)
fi

printf 'Starting optimized vLLM GGUF server:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
