#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/mnt/e/GPTProject2/vLLM}"
WAVES="${EAGER_GRAPH_TUNING_WAVES:-3}"
CONCURRENCY="${EAGER_GRAPH_TUNING_CONCURRENCY:-1,2,4}"
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
  for _ in $(seq 1 240); do
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
  local enforce_eager="$2"
  local max_num_seqs="$3"
  local enforce_flag=()

  if [ "${enforce_eager}" = "1" ]; then
    enforce_flag=(--enforce-eager)
  fi

  local served_model="Qwen3-8B-AWQ-vLLM-local"
  local profile="qwen3_8b_awq_marlin_eager_graph_${profile_id}"
  local output_prefix="reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-${profile_id}-waves${WAVES}"
  local server_log="logs/vllm_eager_graph_tuning_${profile_id}_2026-05-09.log"
  local failure_note="${output_prefix}-startup-failure.txt"

  echo "=== ${profile_id}: enforce_eager=${enforce_eager} max_num_seqs=${max_num_seqs} waves=${WAVES} ==="
  cleanup_vllm

  env \
    VLLM_QUANTIZATION=awq_marlin \
    VLLM_GPU_MEMORY_UTILIZATION=0.85 \
    VLLM_MAX_MODEL_LEN=2048 \
    VLLM_MAX_NUM_SEQS="${max_num_seqs}" \
    VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
    VLLM_ENFORCE_EAGER="${enforce_eager}" \
    VLLM_ENABLE_CHUNKED_PREFILL=1 \
    bash scripts/start_vllm_qwen3_awq_wsl.sh "${WORKSPACE_DIR}" \
      > "${server_log}" 2>&1 &
  local server_pid="$!"

  if ! wait_for_health "${server_pid}" "${server_log}"; then
    {
      echo "profile=${profile}"
      echo "enforce_eager=${enforce_eager}"
      echo "max_num_seqs=${max_num_seqs}"
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
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --block-size 16 \
    --chunked-prefill-enabled \
    "${enforce_flag[@]}" \
    --tokenizer-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
    --output-prefix "${output_prefix}" \
    --notes "eager-graph-tuning-${profile_id}-waves${WAVES}"

  cleanup_vllm
}

trap cleanup_vllm EXIT

run_profile "a-eager-true-util085-seq2" 1 2
run_profile "b-graph-util085-seq2" 0 2
run_profile "d-graph-util085-seq1" 0 1
