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
- Model: `Qwen3-8B-AWQ-vLLM-local`
- Backend: `vllm`
- Profile: `qwen3_8b_awq_marlin_eager_vllm`
- Quantization: `awq-marlin-int4`
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
VLLM_QUANTIZATION=awq_marlin \
VLLM_GPU_MEMORY_UTILIZATION=0.85 \
VLLM_MAX_NUM_SEQS=2 \
VLLM_ENFORCE_EAGER=1 \
bash ./scripts/start_vllm_qwen3_awq_wsl.sh /mnt/e/GPTProject2/vLLM
```

Then run:

```bash
python ./scripts/bench_openai_async.py \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key change-this-before-lan-use \
  --model Qwen3-8B-AWQ-vLLM-local \
  --backend vllm \
  --profile qwen3_8b_awq_marlin_eager_vllm \
  --quantization awq-marlin-int4 \
  --route WSL \
  --workloads short_chat,long_prefill,long_decode,shared_prefix \
  --concurrency 1,2,4,8 \
  --waves 1 \
  --request-rate 0 \
  --streaming \
  --max-model-len 2048 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --block-size 16 \
  --enforce-eager \
  --tokenizer-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ
```

Use `--notes` to explain anything unusual, such as a warm server, a changed GPU memory target, a LAN client route, or background GPU load.

## Stage 3 Prefix-Cache Blocks

PagedAttention / APC experiments use a dedicated runner:

```bash
python ./scripts/bench_prefix_cache_blocks.py \
  --request-count 10 \
  --concurrency 1 \
  --max-model-len 4096 \
  --notes stage3-prefix-cache-blocks
```

It writes:

- `reports/benchmarks/YYYY-MM-DD-vllm-prefix-cache-blocks.csv`
- `reports/benchmarks/YYYY-MM-DD-vllm-prefix-cache-blocks.jsonl`

The default cases are:

| case | shared prefix | purpose |
| --- | ---: | --- |
| `case_a_distinct_1024` | none | Different-prompt control. |
| `case_b_shared_system_1024` | 1024 tokens | System-prefix APC. |
| `case_c_shared_document_2048` | 2048 tokens | Long-document APC. |

This runner records `expected_shared_blocks`, `/metrics` prefix-cache query/hit deltas, and server-side prefill/decode metric deltas when vLLM exposes them. See `docs/PAGED_ATTENTION_PREFIX_CACHE.md` for the full contract.

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
