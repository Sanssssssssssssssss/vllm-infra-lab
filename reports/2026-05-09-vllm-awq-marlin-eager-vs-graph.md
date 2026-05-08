# 2026-05-09 AWQ-Marlin Eager vs CUDA Graph

## Goal

Decide whether `enforce_eager=true` is only a startup workaround or the real
best serving profile for `Qwen3-8B-AWQ + awq_marlin`.

The tested baseline shape kept the latest prefill decision:

```text
max_model_len = 2048
gpu_memory_utilization = 0.85
max_num_batched_tokens = 4096
chunked_prefill = on
prefix_caching = on
async_scheduling = on
```

## Startup Results

| group | enforce_eager | gpu_util | max_num_seqs | status | CUDA graph | GPU KV tokens | max concurrency | GPU used after start |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| A | true | 0.85 | 2 | ready | false | 3328 | 1.62x | 7380 MB |
| B | false | 0.85 | 2 | timeout at 420s | false |  |  | 7841 MB |
| B retry | false | 0.85 | 2 | ready | true | 3600 | 1.76x | 7332 MB |
| C | false | 0.83 | 2 | failed | false |  |  | 6590 MB |
| D | false | 0.85 | 1 | ready | true | 3168 | 1.55x | 7132 MB |

The first B attempt timed out during cold graph/compile setup. A retry with a
longer startup window completed successfully and captured CUDA graphs. This
means graph mode is not a pure startup failure, but it has a cold-start caveat.

C failed because available KV cache was `0.27 GiB`, just below the `0.28 GiB`
needed to serve one `2048`-token request.

## Request Benchmark

Benchmarked profiles used all four workloads at concurrency `1,2,4` with
`waves=3`.

### Long Decode

| profile | c | TTFT p95 ms | ITL p95 ms | E2E p95 ms | output TPS | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A eager seq2 | 1 | 69.94 | 35.05 | 15582.83 | 32.93 | 0 |
| B graph seq2 | 1 | 49.45 | 30.28 | 14284.29 | 35.91 | 0 |
| D graph seq1 | 1 | 47.36 | 30.27 | 14266.93 | 35.96 | 0 |
| A eager seq2 | 2 | 91.04 | 36.53 | 16013.32 | 64.41 | 0 |
| B graph seq2 | 2 | 80.44 | 30.84 | 14555.96 | 70.45 | 0 |
| D graph seq1 | 2 | 14466.14 | 30.48 | 28771.89 | 35.78 | 0 |
| A eager seq2 | 4 | 15910.58 | 35.41 | 31572.07 | 65.15 | 0 |
| B graph seq2 | 4 | 14640.23 | 30.84 | 29101.33 | 70.46 | 0 |
| D graph seq1 | 4 | 43074.40 | 30.22 | 57277.70 | 35.85 | 0 |

### Short Chat

| profile | c | TTFT p95 ms | ITL p95 ms | output TPS | errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| A eager seq2 | 1 | 325.44 | 32.96 | 32.95 | 0 |
| B graph seq2 | 1 | 295.22 | 33.19 | 34.76 | 0 |
| D graph seq1 | 1 | 327.15 | 31.32 | 34.18 | 0 |
| A eager seq2 | 2 | 93.31 | 36.92 | 65.44 | 0 |
| B graph seq2 | 2 | 71.81 | 30.02 | 71.44 | 0 |
| D graph seq1 | 2 | 3600.59 | 30.07 | 36.05 | 0 |
| A eager seq2 | 4 | 4086.03 | 39.41 | 63.34 | 0 |
| B graph seq2 | 4 | 3682.26 | 30.39 | 70.91 | 0 |
| D graph seq1 | 4 | 10699.15 | 30.11 | 36.11 | 0 |

## Decision

Switch the active AWQ-Marlin default to CUDA graph mode:

```text
enforce_eager = false
max_num_seqs = 2
gpu_memory_utilization = 0.85
max_num_batched_tokens = 4096
```

Reasoning:

- Graph seq2 had `error_count=0` across the request benchmark.
- Graph seq2 improved `long_decode` ITL p95 at c1/c2/c4.
- Graph seq2 improved c2/c4 output TPS and short-chat TTFT p95 versus eager.
- Graph seq1 is not a default candidate: it improves per-token latency but
  collapses c2/c4 queueing and aggregate throughput.
- Graph seq2 is not more memory-fragile than eager in the retry: it started with
  `3600` GPU KV tokens and `7332 MB` used after startup, versus eager's `3328`
  GPU KV tokens and `7380 MB`.

Operational caveat: the first graph seq2 startup timed out at 420 seconds during
cold compile/cache setup. Keep `VLLM_ENFORCE_EAGER=1` as an explicit fallback,
and allow a longer first-start window when CUDA graph caches are cold.

## Artifacts

```text
reports/memory/2026-05-09-vllm-awq-marlin-eager-graph-startup.csv
reports/memory/2026-05-09-vllm-awq-marlin-eager-graph-startup.jsonl
reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-a-eager-true-util085-seq2-waves3.csv
reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-a-eager-true-util085-seq2-waves3.jsonl
reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-b-graph-util085-seq2-waves3.csv
reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-b-graph-util085-seq2-waves3.jsonl
reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-d-graph-util085-seq1-waves3.csv
reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-d-graph-util085-seq1-waves3.jsonl
```
