# Benchmarking

This project treats benchmark output as a first-class artifact. Do not tune vLLM parameters from smoke tests, one-off chat replies, or manual log watching alone.

## Runner

Use the OpenAI-compatible async runner:

```bash
cd /mnt/e/GPTProject2/vLLM
source ~/.venvs/gptproject2-vllm/bin/activate
python ./scripts/bench_openai_async.py
```

Default target:

- Endpoint: `http://127.0.0.1:8000/v1/chat/completions`
- Model: `Qwen3-8B-GGUF-vLLM-local`
- Backend: `vllm`
- Profile: `qwen3_8b_gguf_vllm_optimized`
- Quantization: `gguf-q4_k_m`
- Route: `WSL`

The runner writes:

- `reports/benchmarks/YYYY-MM-DD-vllm-gguf-matrix.csv`
- `reports/benchmarks/YYYY-MM-DD-vllm-gguf-matrix.jsonl`

CSV is for spreadsheet-style comparisons. JSONL keeps request-level detail in each matrix row.

## Required Workloads

| workload | prompt | output | purpose |
| --- | ---: | ---: | --- |
| `short_chat` | 128 tokens | 128 tokens | Basic chat performance |
| `long_prefill` | 1024 tokens | 32 tokens | Prefill and TTFT behavior |
| `long_decode` | 128 tokens | 512 tokens | Decode and ITL behavior |
| `shared_prefix` | 1024 shared-prefix tokens | 32 tokens | Automatic prefix caching behavior |

Run each workload at:

```text
concurrency = 1, 2, 4, 8
```

The default command runs that full first matrix with one wave per concurrency level. That means total requests per row equals `concurrency * waves`.

## Required Fields

Every benchmark row must include at least:

```text
run_id
timestamp
backend
profile
model
quantization
vllm_version
gpu_name
vram_total
route
prompt_tokens
output_tokens
max_model_len
max_num_seqs
max_num_batched_tokens
gpu_memory_utilization
block_size
prefix_caching_enabled
chunked_prefill_enabled
async_scheduling_enabled
enforce_eager
streaming
concurrency
request_rate
ttft_ms_p50
ttft_ms_p95
itl_ms_p50
itl_ms_p95
e2e_latency_ms_p50
e2e_latency_ms_p95
output_tps
total_tps
gpu_mem_used_mb
error_count
notes
```

The current runner also records `workload`, target token counts, request counts, success counts, total token counts, per-request JSON details, wall time, and workload purpose.

## Metric Meaning

- `TTFT`: time to first token. This is mainly affected by prefill, queueing, prefix cache, and chunked prefill.
- `ITL`: inter-token latency. This is the clearest user-facing decode smoothness signal.
- `output_tps`: aggregate output tokens per second across successful requests in the row.
- `total_tps`: aggregate prompt plus output tokens per second.
- `E2E latency`: full request latency from client send to stream completion.
- `gpu_mem_used_mb`: GPU memory after the row completes. Use this with vLLM logs for KV capacity and OOM boundaries.
- `error_count`: count of failed requests in that row. OOM, timeout, HTTP error, and preemption-like failures belong here and in `notes`.

TTFT and ITL require `streaming=true`. If a backend is tested without streaming, TTFT and ITL should be blank and the result must say so in `notes`.

## Baseline Command

```bash
python ./scripts/bench_openai_async.py \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key change-this-before-lan-use \
  --model Qwen3-8B-GGUF-vLLM-local \
  --backend vllm \
  --profile qwen3_8b_gguf_vllm_optimized \
  --quantization gguf-q4_k_m \
  --route WSL \
  --workloads short_chat,long_prefill,long_decode,shared_prefix \
  --concurrency 1,2,4,8 \
  --waves 1 \
  --request-rate 0 \
  --streaming
```

Use `--notes` to explain anything unusual, such as a warm server, a changed GPU memory target, a LAN client route, or background GPU load.

## First Recorded Matrix

The first recorded matrix is:

- `reports/benchmarks/2026-05-05-vllm-gguf-matrix.csv`
- `reports/benchmarks/2026-05-05-vllm-gguf-matrix.jsonl`

It covers the four required workloads at concurrency `1,2,4,8` with `waves=1` and `error_count=0` across all matrix rows.

## Rules

- Run at least a small benchmark after changing benchmark code or launch parameters.
- Push each completed benchmark or benchmark-infra change to the GitHub repository before moving on.
- Never commit model weights, Hugging Face cache, runtime API keys, Python virtualenvs, or raw server logs.
- Keep `reports/benchmarks/*.csv` and `reports/benchmarks/*.jsonl` append-only unless explicitly correcting a bad run.
