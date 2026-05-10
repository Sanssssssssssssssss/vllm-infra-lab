#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/mnt/e/GPTProject2/vLLM}"
DATE_TAG="${SPEC_DECODE_DATE_TAG:-$(date +%F)}"
WAVES="${SPEC_DECODE_WAVES:-3}"
CONCURRENCY="${SPEC_DECODE_CONCURRENCY:-1,2,4}"
WORKLOADS="${SPEC_DECODE_WORKLOADS:-long_decode}"
VARIANTS="${SPEC_DECODE_VARIANTS:-baseline,baseline_sync,ngram4_sync,ngram8_sync,suffix_sync}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/gptproject2-vllm}"

cd "${WORKSPACE_DIR}"
source "${VENV_DIR}/bin/activate"

mkdir -p logs reports/benchmarks reports/metrics

cleanup_vllm() {
  pkill -f "vllm serve ${WORKSPACE_DIR}/models/Qwen3-8B-AWQ" >/dev/null 2>&1 || true
  sleep 4
}

wait_for_health() {
  local server_pid="$1"
  local log_path="$2"
  for _ in $(seq 1 240); do
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

capture_metrics() {
  local path="$1"
  curl -fsS "http://127.0.0.1:8000/metrics" > "${path}" 2>/dev/null || true
}

spec_config_for_variant() {
  case "$1" in
    baseline|baseline_sync)
      printf ''
      ;;
    ngram4|ngram4_sync)
      printf '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":1}'
      ;;
    ngram8|ngram8_sync)
      printf '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8,"prompt_lookup_min":1}'
      ;;
    suffix|suffix_sync)
      printf '{"method":"suffix","suffix_decoding_max_tree_depth":8,"suffix_decoding_max_cached_requests":10000,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}'
      ;;
    *)
      echo "Unknown speculative decoding variant: $1" >&2
      return 2
      ;;
  esac
}

async_scheduling_for_variant() {
  case "$1" in
    baseline)
      printf '1'
      ;;
    *)
      # vLLM 0.17.1 rejects async scheduling with n-gram and suffix proposers.
      printf '0'
      ;;
  esac
}

run_variant() {
  local variant="$1"
  local spec_config
  spec_config="$(spec_config_for_variant "${variant}")"
  local async_scheduling
  async_scheduling="$(async_scheduling_for_variant "${variant}")"
  local async_flag=(--no-async-scheduling-enabled)
  if [ "${async_scheduling}" = "1" ]; then
    async_flag=(--async-scheduling-enabled)
  fi
  local served_model="Qwen3-8B-AWQ-vLLM-local"
  local profile="qwen3_8b_awq_marlin_specdecode_${variant}"
  local output_prefix="reports/benchmarks/${DATE_TAG}-vllm-awq-marlin-specdecode-${variant}-waves${WAVES}"
  local metrics_prefix="reports/benchmarks/${DATE_TAG}-vllm-awq-marlin-specdecode-metrics"
  local before_metrics="reports/metrics/${DATE_TAG}-vllm-awq-marlin-specdecode-${variant}-before.prom"
  local after_metrics="reports/metrics/${DATE_TAG}-vllm-awq-marlin-specdecode-${variant}-after.prom"
  local server_log="logs/vllm_specdecode_${variant}_${DATE_TAG}.log"
  local failure_note="${output_prefix}-startup-failure.txt"
  local env_args=()

  echo "=== speculative decoding variant=${variant} async_scheduling=${async_scheduling} waves=${WAVES} concurrency=${CONCURRENCY} ==="
  cleanup_vllm

  if [ -n "${spec_config}" ]; then
    env_args+=(VLLM_SPECULATIVE_CONFIG="${spec_config}")
  fi

  env \
    VLLM_QUANTIZATION=awq_marlin \
    VLLM_GPU_MEMORY_UTILIZATION=0.85 \
    VLLM_MAX_MODEL_LEN=2048 \
    VLLM_MAX_NUM_SEQS=2 \
    VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
    VLLM_BLOCK_SIZE=16 \
    VLLM_ENFORCE_EAGER=0 \
    VLLM_ENABLE_PREFIX_CACHING=1 \
    VLLM_ENABLE_CHUNKED_PREFILL=1 \
    VLLM_ASYNC_SCHEDULING="${async_scheduling}" \
    VLLM_KV_CACHE_METRICS=1 \
    "${env_args[@]}" \
    bash scripts/start_vllm_qwen3_awq_wsl.sh "${WORKSPACE_DIR}" \
      > "${server_log}" 2>&1 &
  local server_pid="$!"

  if ! wait_for_health "${server_pid}" "${server_log}"; then
    {
      echo "profile=${profile}"
      echo "variant=${variant}"
      echo "async_scheduling=${async_scheduling}"
      echo "speculative_config=${spec_config}"
      echo "status=startup_failed"
      echo "server_log=${server_log}"
      tail -n 200 "${server_log}" || true
    } > "${failure_note}"
    cleanup_vllm
    return 0
  fi

  capture_metrics "${before_metrics}"

  python ./scripts/bench_openai_async.py \
    --host 127.0.0.1 \
    --port 8000 \
    --api-key change-this-before-lan-use \
    --model "${served_model}" \
    --backend vllm \
    --profile "${profile}" \
    --quantization awq-marlin-int4 \
    --route WSL \
    --workloads "${WORKLOADS}" \
    --concurrency "${CONCURRENCY}" \
    --waves "${WAVES}" \
    --request-rate 0 \
    --streaming \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --block-size 16 \
    --chunked-prefill-enabled \
    "${async_flag[@]}" \
    --tokenizer-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
    --output-prefix "${output_prefix}" \
    --notes "speculative-decoding-${variant};async_scheduling=${async_scheduling}"

  capture_metrics "${after_metrics}"

  python ./scripts/extract_spec_decode_metrics.py \
    --before "${before_metrics}" \
    --after "${after_metrics}" \
    --variant "${variant}" \
    --profile "${profile}" \
    --speculative-config "${spec_config}" \
    --output-prefix "${metrics_prefix}" \
    --notes "speculative-decoding-${variant};async_scheduling=${async_scheduling}"

  cleanup_vllm
}

trap cleanup_vllm EXIT

IFS=',' read -r -a variant_list <<< "${VARIANTS}"
for variant in "${variant_list[@]}"; do
  run_variant "${variant}"
done
