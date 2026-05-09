#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/mnt/e/GPTProject2/vLLM}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/gptproject2-vllm}"
DATE_TAG="${KV_DTYPE_DATE_TAG:-$(date +%F)}"
VARIANTS="${KV_DTYPE_QUALITY_VARIANTS:-auto:TRITON_ATTN,fp8_e4m3:TRITON_ATTN}"
MAX_MODEL_LEN="${KV_DTYPE_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${KV_DTYPE_MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${KV_DTYPE_MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${KV_DTYPE_GPU_MEMORY_UTILIZATION:-0.85}"
ENFORCE_EAGER="${KV_DTYPE_ENFORCE_EAGER:-1}"

cd "${WORKSPACE_DIR}"
source "${VENV_DIR}/bin/activate"

mkdir -p logs reports/quality

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
  local kv_dtype="$1"
  local attention_backend="$2"
  local backend_slug
  backend_slug="$(printf '%s' "${attention_backend}" | tr '[:upper:]' '[:lower:]')"
  local profile="qwen3_8b_awq_marlin_kv_${kv_dtype}_${backend_slug}_quality"
  local output_prefix="reports/quality/${DATE_TAG}-vllm-awq-marlin-kv-quality-${kv_dtype}-${backend_slug}"
  local server_log="logs/vllm_kv_dtype_quality_${kv_dtype}_${backend_slug}_${DATE_TAG}.log"
  local eager_flag=()

  if [ "${ENFORCE_EAGER}" = "1" ]; then
    eager_flag=(--enforce-eager)
  fi

  echo "=== KV dtype quality: kv=${kv_dtype} backend=${attention_backend} ==="
  cleanup_vllm

  env \
    VLLM_KV_CACHE_DTYPE="${kv_dtype}" \
    VLLM_ATTENTION_BACKEND="${attention_backend}" \
    VLLM_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    VLLM_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    VLLM_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    VLLM_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    VLLM_ENFORCE_EAGER="${ENFORCE_EAGER}" \
    bash scripts/start_vllm_qwen3_awq_wsl.sh "${WORKSPACE_DIR}" \
      > "${server_log}" 2>&1 &
  local server_pid="$!"

  if ! wait_for_health "${server_pid}" "${server_log}"; then
    {
      echo "profile=${profile}"
      echo "kv_cache_dtype=${kv_dtype}"
      echo "attention_backend=${attention_backend}"
      echo "status=startup_failed"
      echo "server_log=${server_log}"
      tail -n 200 "${server_log}" || true
    } > "${output_prefix}-startup-failure.txt"
    cleanup_vllm
    return 0
  fi

  set +e
  python ./scripts/quality_regression_openai.py \
    --host 127.0.0.1 \
    --port 8000 \
    --api-key change-this-before-lan-use \
    --model Qwen3-8B-AWQ-vLLM-local \
    --backend vllm \
    --profile "${profile}" \
    --quantization awq-marlin-int4 \
    --kv-cache-dtype "${kv_dtype}" \
    --route WSL \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --block-size 16 \
    --attention-backend "${attention_backend}" \
    --prefix-caching-enabled \
    --chunked-prefill-enabled \
    --async-scheduling-enabled \
    "${eager_flag[@]}" \
    --output-prefix "${output_prefix}" \
    --notes "kv-cache-dtype-quality-${kv_dtype}-${backend_slug}"
  local quality_exit="$?"
  set -e

  echo "quality_exit=${quality_exit}" > "${output_prefix}.status"
  cleanup_vllm
}

trap cleanup_vllm EXIT

IFS=',' read -ra variant_list <<< "${VARIANTS}"
for variant in "${variant_list[@]}"; do
  kv_dtype="${variant%%:*}"
  attention_backend="${variant#*:}"
  run_variant "${kv_dtype}" "${attention_backend}"
done
