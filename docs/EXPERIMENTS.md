# Experiments

This file is the experiment contract for the local inference lab. It keeps tuning work from drifting into undocumented one-off changes.

## Current Baseline

The active vLLM optimization baseline is now:

- Backend: `vllm`
- Route: `WSL`
- Profile: `qwen3_8b_awq_marlin_eager_vllm`
- Model: `Qwen3-8B-AWQ-vLLM-local`
- Quantization: `awq-marlin-int4`
- `max_model_len=2048`
- `max_num_seqs=2`
- `max_num_batched_tokens=4096`
- `gpu_memory_utilization=0.85`
- `block_size=16`
- Prefix caching: on
- Chunked prefill: on
- Async scheduling: on
- Enforce eager: on
- Streaming benchmark: on

The previous GGUF baseline remains available through
`scripts/start_vllm_gguf_optimized_wsl.sh`. The quantized-model selection notes
live in `docs/QUANTIZATION_MODELS.md`.

Latest boundary baseline:

- `reports/2026-05-08-vllm-awq-marlin-boundary-waves5.md`
- `reports/benchmarks/2026-05-08-vllm-awq-marlin-boundary-waves5.csv`
- `reports/benchmarks/2026-05-08-vllm-awq-marlin-boundary-waves5.jsonl`

This run completed the required four-workload matrix at concurrency `1,2,4,8`
with `waves=5`. Stability passed, but agent-facing TTFT p95 showed the current
interactive boundary is closer to concurrency `2`.

Latest AWQ-Marlin startup memory sweep:

- `reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.md`
- `reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.csv`
- `reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.jsonl`
- `reports/memory/2026-05-08-vllm-awq-marlin-candidate-restarts.csv`
- `reports/memory/2026-05-08-vllm-awq-marlin-candidate-restarts.jsonl`

This startup-only sweep retained `0.85` as the highest useful
`gpu_memory_utilization` for AWQ-Marlin on the 8GB GPU. `0.86` and `0.88`
were OOM-style startup boundaries in this run.

Latest prefill tuning pass:

- `reports/2026-05-08-vllm-awq-marlin-prefill-tuning-waves3.md`
- `reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-a-batch2048-chunked-on-waves3.csv`
- `reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-b-batch4096-chunked-on-waves3.csv`
- `reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-d-batch2048-chunked-off-waves3.csv`
- `reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-e-batch4096-chunked-off-waves3.csv`

Decision: keep `max_num_batched_tokens=4096` and chunked prefill enabled for
the active AWQ-Marlin route. `8192` failed startup on this 8GB GPU, and chunked
prefill off did not show a clear cross-workload advantage.

## Experiment Loop

1. Change one parameter family at a time.
2. Start or restart the backend with the exact intended profile.
3. Run the required benchmark matrix from `docs/BENCHMARKING.md`.
4. Record CSV and JSONL artifacts under `reports/benchmarks/`.
5. Update notes when the run is not clean.
6. Commit and push the benchmark script, docs, and result artifacts before continuing.

## First Matrix

The first required matrix is:

```text
workloads = short_chat, long_prefill, long_decode, shared_prefix
concurrency = 1, 2, 4, 8
waves = 1 or higher
request_rate = 0 for burst unless testing arrival shaping
```

This matrix is enough to separate the first-order effects:

- `short_chat`: basic interactive behavior.
- `long_prefill`: prefill pressure and TTFT.
- `long_decode`: decode pressure and ITL.
- `shared_prefix`: automatic prefix caching.

## Planned Axes

## Stage 2: Memory And KV Cache

Before deeper request-level tuning, profile startup memory and KV cache capacity. The full Stage 2 matrix is documented in `docs/MEMORY_MODEL.md` and is run with:

```bash
python ./scripts/profile_vllm_memory_sweep.py \
  --preset stage2 \
  --confirm-large-sweep \
  --kill-existing \
  --notes stage2-full-memory-sweep
```

This stage records model memory, KV cache memory, derived GPU blocks, GPU KV tokens, max concurrency, CUDA graph capture, and OOM status.

Run the pilot first after changing the profiler:

```bash
python ./scripts/profile_vllm_memory_sweep.py \
  --preset pilot \
  --kill-existing \
  --notes stage2-memory-pilot
```

Only run the full 288-start matrix after the pilot produces clean CSV/JSONL rows and the machine can be left alone.

## Stage 3: PagedAttention Prefix Blocks

The Stage 3 contract is documented in `docs/PAGED_ATTENTION_PREFIX_CACHE.md`.
It isolates Automatic Prefix Caching as a KV block reuse problem.

Default run:

```bash
python ./scripts/bench_prefix_cache_blocks.py \
  --request-count 10 \
  --concurrency 1 \
  --max-model-len 4096 \
  --notes stage3-prefix-cache-blocks
```

Required cases:

```text
Case A: 10 requests with different 1024-token prompts
Case B: 10 requests sharing a 1024-token system prefix
Case C: 10 requests sharing a 2048-token document prefix
```

With `block_size=16`, the expected reusable full-block counts are 64 for Case B
and 128 for Case C. Use `prefix_cache_hits_delta`, `prefix_cache_hit_rate`,
`ttft_ms_p50/p95`, and `request_prefill_time_seconds_sum_delta` to decide
whether APC is actually active. `output_tps` should be treated as a guardrail:
it should remain roughly stable because prefix caching reduces prefill work, not
decode work.

### Prefill

Vary:

```text
max_num_batched_tokens = 2048, 4096, 8192
chunked_prefill_enabled = true, false
```

Watch:

- TTFT p50/p95
- E2E p50/p95
- error_count
- GPU memory after long-prefill rows

### Continuous Batching

Vary:

```text
max_num_seqs = 1, 2, 4, 8
concurrency = 1, 2, 4, 8
```

Watch:

- aggregate output_tps
- E2E p95 under concurrency
- queueing symptoms through TTFT p95
- error_count

### KV Cache

Vary:

```text
max_model_len = 1024, 2048, 4096
block_size = 16, 32
kv_cache_dtype = auto, fp8_e5m2
```

Watch:

- available KV capacity from vLLM startup logs
- long context success/failure boundary
- output quality when fp8 KV cache is used
- OOM and preemption-like errors

### Prefix Caching

Vary:

```text
prefix_caching_enabled = true, false
prefix_caching_hash_algo = sha256, xxhash
```

Watch:

- shared_prefix TTFT p50/p95
- prefix cache hit rate from vLLM metrics/logs
- CPU overhead if hash algorithm becomes visible at high request volume

### CUDA Graph / Eager Control

Vary:

```text
enforce_eager = false, true
```

Watch:

- ITL p50/p95
- startup time
- memory headroom
- compatibility failures

## Result Interpretation

Prefer configurations that improve the workload they target without causing p95 latency spikes or new errors elsewhere. A higher aggregate TPS is not automatically better if TTFT p95 or error_count gets worse for interactive workloads.

For this laptop, a configuration is considered unstable if it:

- requires lowering Windows/WSL background GPU usage by hand to start,
- produces OOM or timeout in any required row,
- leaves less than a small practical VRAM margin after startup,
- improves one workload only by severely degrading another required workload.
