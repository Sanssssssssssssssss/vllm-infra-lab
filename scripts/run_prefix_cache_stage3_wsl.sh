#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/mnt/e/GPTProject2/vLLM}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/gptproject2-vllm}"
DATE_TAG="${STAGE3_DATE_TAG:-$(date +%F)}"
VARIANTS="${STAGE3_PREFIX_VARIANTS:-16:sha256,32:sha256,16:xxhash,32:xxhash}"
CASES="${STAGE3_CASES:-case_a_distinct_1024,case_b_shared_system_1024,case_c_shared_document_2048}"
REQUEST_COUNT="${STAGE3_REQUEST_COUNT:-10}"
CONCURRENCY="${STAGE3_CONCURRENCY:-1}"
MAX_MODEL_LEN="${STAGE3_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${STAGE3_MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${STAGE3_MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${STAGE3_GPU_MEMORY_UTILIZATION:-0.85}"
ENFORCE_EAGER="${STAGE3_ENFORCE_EAGER:-1}"

cd "${WORKSPACE_DIR}"
source "${VENV_DIR}/bin/activate"

mkdir -p logs reports/benchmarks

cleanup_vllm() {
  pkill -f "vllm serve ${WORKSPACE_DIR}/models/Qwen3-8B-AWQ" >/dev/null 2>&1 || true
  sleep 4
}

wait_for_health() {
  local server_pid="$1"
  local log_path="$2"
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "vLLM exited before healthcheck."
      tail -n 160 "${log_path}" || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for vLLM healthcheck."
  tail -n 200 "${log_path}" || true
  return 1
}

run_variant() {
  local block_size="$1"
  local hash_algo="$2"
  local profile_id="awq-marlin-stage3-prefix-block${block_size}-${hash_algo}"
  local profile="qwen3_8b_awq_marlin_stage3_prefix_blocks_block${block_size}_${hash_algo}"
  local output_prefix="reports/benchmarks/${DATE_TAG}-vllm-awq-marlin-prefix-cache-block${block_size}-${hash_algo}"
  local server_log="logs/vllm_stage3_prefix_blocks_block${block_size}_${hash_algo}_${DATE_TAG}.log"
  local failure_note="${output_prefix}-startup-failure.txt"
  local eager_flag=()

  if [ "${ENFORCE_EAGER}" = "1" ]; then
    eager_flag=(--enforce-eager)
  fi

  echo "=== Stage 3 APC: block_size=${block_size} hash=${hash_algo} ==="
  cleanup_vllm

  env \
    STAGE3_VLLM_LOG_PATH="${server_log}" \
    VLLM_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    VLLM_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    VLLM_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    VLLM_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    VLLM_BLOCK_SIZE="${block_size}" \
    VLLM_PREFIX_CACHING_HASH_ALGO="${hash_algo}" \
    VLLM_KV_CACHE_METRICS=1 \
    VLLM_ENABLE_PREFIX_CACHING=1 \
    VLLM_ENABLE_CHUNKED_PREFILL=1 \
    VLLM_ASYNC_SCHEDULING=1 \
    VLLM_ENFORCE_EAGER="${ENFORCE_EAGER}" \
    bash scripts/start_vllm_stage3_prefix_blocks_wsl.sh "${WORKSPACE_DIR}" &
  local server_pid="$!"

  if ! wait_for_health "${server_pid}" "${server_log}"; then
    {
      echo "profile=${profile}"
      echo "block_size=${block_size}"
      echo "prefix_caching_hash_algo=${hash_algo}"
      echo "status=startup_failed"
      echo "server_log=${server_log}"
      tail -n 200 "${server_log}" || true
    } > "${failure_note}"
    cleanup_vllm
    return 0
  fi

  python ./scripts/bench_prefix_cache_blocks.py \
    --host 127.0.0.1 \
    --port 8000 \
    --api-key change-this-before-lan-use \
    --model Qwen3-8B-AWQ-vLLM-local \
    --backend vllm \
    --profile "${profile}" \
    --quantization awq-marlin-int4 \
    --route WSL \
    --cases "${CASES}" \
    --request-count "${REQUEST_COUNT}" \
    --concurrency "${CONCURRENCY}" \
    --request-rate 0 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --block-size "${block_size}" \
    --prefix-caching-hash-algo "${hash_algo}" \
    --prefix-caching-enabled \
    --chunked-prefill-enabled \
    --async-scheduling-enabled \
    "${eager_flag[@]}" \
    --tokenizer-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
    --output-prefix "${output_prefix}" \
    --notes "${profile_id}"

  cleanup_vllm
}

trap cleanup_vllm EXIT

IFS=',' read -ra variant_list <<< "${VARIANTS}"
for variant in "${variant_list[@]}"; do
  block_size="${variant%%:*}"
  hash_algo="${variant#*:}"
  run_variant "${block_size}" "${hash_algo}"
done
