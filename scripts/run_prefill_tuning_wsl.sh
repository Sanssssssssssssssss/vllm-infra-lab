#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/mnt/e/GPTProject2/vLLM}"
WAVES="${PREFILL_TUNING_WAVES:-3}"
CONCURRENCY="${PREFILL_TUNING_CONCURRENCY:-1,2,4,8}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/gptproject2-vllm}"

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
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "vLLM exited before healthcheck."
      tail -n 120 "${log_path}" || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for vLLM healthcheck."
  tail -n 160 "${log_path}" || true
  return 1
}

run_profile() {
  local profile_id="$1"
  local batched_tokens="$2"
  local chunked="$3"
  local chunk_flag="--chunked-prefill-enabled"
  local chunk_env="1"
  local chunk_label="on"

  if [ "${chunked}" = "off" ]; then
    chunk_flag="--no-chunked-prefill-enabled"
    chunk_env="0"
    chunk_label="off"
  fi

  local served_model="Qwen3-8B-AWQ-vLLM-local"
  local profile="qwen3_8b_awq_marlin_prefill_${profile_id}"
  local output_prefix="reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-${profile_id}-waves${WAVES}"
  local server_log="logs/vllm_prefill_tuning_${profile_id}_2026-05-08.log"
  local failure_note="${output_prefix}-startup-failure.txt"

  echo "=== ${profile_id}: batched=${batched_tokens} chunked=${chunk_label} waves=${WAVES} ==="
  cleanup_vllm

  env \
    VLLM_QUANTIZATION=awq_marlin \
    VLLM_GPU_MEMORY_UTILIZATION=0.85 \
    VLLM_MAX_MODEL_LEN=2048 \
    VLLM_MAX_NUM_SEQS=2 \
    VLLM_MAX_NUM_BATCHED_TOKENS="${batched_tokens}" \
    VLLM_ENFORCE_EAGER=1 \
    VLLM_ENABLE_CHUNKED_PREFILL="${chunk_env}" \
    bash scripts/start_vllm_qwen3_awq_wsl.sh "${WORKSPACE_DIR}" \
      > "${server_log}" 2>&1 &
  local server_pid="$!"

  if ! wait_for_health "${server_pid}" "${server_log}"; then
    {
      echo "profile=${profile}"
      echo "max_num_batched_tokens=${batched_tokens}"
      echo "chunked_prefill=${chunk_label}"
      echo "status=startup_failed"
      echo "server_log=${server_log}"
      tail -n 160 "${server_log}" || true
    } > "${failure_note}"
    cleanup_vllm
    return 0
  fi

  python ./scripts/bench_openai_async.py \
    --host 127.0.0.1 \
    --port 8000 \
    --api-key change-this-before-lan-use \
    --model "${served_model}" \
    --backend vllm \
    --profile "${profile}" \
    --quantization awq-marlin-int4 \
    --route WSL \
    --workloads short_chat,long_prefill,long_decode,shared_prefix \
    --concurrency "${CONCURRENCY}" \
    --waves "${WAVES}" \
    --request-rate 0 \
    --streaming \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    --max-num-batched-tokens "${batched_tokens}" \
    --gpu-memory-utilization 0.85 \
    --block-size 16 \
    --enforce-eager \
    --tokenizer-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
    "${chunk_flag}" \
    --output-prefix "${output_prefix}" \
    --notes "prefill-tuning-${profile_id}-batch${batched_tokens}-chunked-${chunk_label}-waves${WAVES}"

  cleanup_vllm
}

trap cleanup_vllm EXIT

run_profile "a-batch2048-chunked-on" 2048 on
run_profile "b-batch4096-chunked-on" 4096 on
run_profile "c-batch8192-chunked-on" 8192 on
run_profile "d-batch2048-chunked-off" 2048 off
run_profile "e-batch4096-chunked-off" 4096 off
