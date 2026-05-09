# 2026-05-09 AWQ-Marlin Prefix Cache Blocks

## Goal

Confirm that Automatic Prefix Caching is actually reusing KV blocks and reducing
prefill/TTFT, not just enabled as a launch flag.

The benchmark used the Stage 3 cases:

```text
Case A: 10 requests with distinct 1024-token prompts
Case B: 10 requests sharing a 1024-token system prefix
Case C: 10 requests sharing a 2048-token document prefix
```

## Runtime Shape

The AWQ-Marlin 4096-context Stage 3 shape was:

```text
model = Qwen3-8B-AWQ-vLLM-local
quantization = awq-marlin-int4
max_model_len = 4096
max_num_seqs = 1
max_num_batched_tokens = 2048
gpu_memory_utilization = 0.85
prefix_caching = on
chunked_prefill = on
async_scheduling = on
kv_cache_metrics = on
enforce_eager = true
```

This deliberately uses eager mode because the 4096-token AWQ profile is right on
the 8GB VRAM KV boundary. The active interactive AWQ profile remains CUDA graph
for 2048-token serving; this Stage 3 profile trades that off to make Case C fit.

One `block_size=32, sha256` startup attempt failed with available KV cache
`0.55 GiB` versus `0.56 GiB` required for 4096 tokens. A same-parameter retry
started and completed the benchmark, so this is recorded as a memory-edge caveat,
not a failed benchmark row.

## Results

| block | hash | case | expected blocks | TTFT p50 ms | TTFT p95 ms | prefill sum s | prefix hits | hit rate | output TPS | errors |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | sha256 | distinct 1024 | 0 | 486.20 | 664.79 | 5.0183 | 0 | 0.0000 | 22.57 | 0 |
| 16 | sha256 | shared system 1024 | 64 | 54.50 | 64.48 | 0.4665 | 10240 | 0.9632 | 33.53 | 0 |
| 16 | sha256 | shared document 2048 | 128 | 58.50 | 65.60 | 0.4618 | 20640 | 0.9880 | 30.59 | 0 |
| 32 | sha256 | distinct 1024 | 0 | 498.22 | 771.10 | 5.3528 | 0 | 0.0000 | 21.75 | 0 |
| 32 | sha256 | shared system 1024 | 32 | 57.57 | 60.81 | 0.4679 | 10240 | 0.9632 | 32.98 | 0 |
| 32 | sha256 | shared document 2048 | 64 | 55.23 | 62.98 | 0.4554 | 20480 | 0.9803 | 30.53 | 0 |
| 16 | xxhash | distinct 1024 | 0 | 491.01 | 681.30 | 5.1003 | 0 | 0.0000 | 22.52 | 0 |
| 16 | xxhash | shared system 1024 | 64 | 53.68 | 64.63 | 0.4583 | 10240 | 0.9632 | 32.67 | 0 |
| 16 | xxhash | shared document 2048 | 128 | 61.11 | 72.33 | 0.5128 | 20640 | 0.9880 | 30.01 | 0 |
| 32 | xxhash | distinct 1024 | 0 | 501.59 | 720.90 | 5.2816 | 0 | 0.0000 | 21.77 | 0 |
| 32 | xxhash | shared system 1024 | 32 | 58.04 | 64.82 | 0.4726 | 10240 | 0.9632 | 33.18 | 0 |
| 32 | xxhash | shared document 2048 | 64 | 57.33 | 63.12 | 0.4591 | 20480 | 0.9803 | 30.38 | 0 |

## Interpretation

APC is active:

- Distinct prompts produced `0` prefix cache hits and paid about `5.0-5.35s`
  total prefill time across the 10 measured requests.
- Shared-prefix cases produced large hit deltas: `10240` for Case B and
  `20480-20640` for Case C.
- Shared-prefix hit rates were `0.9632` for Case B and `0.9803-0.9880` for
  Case C.
- TTFT p50 dropped from about `486-502 ms` to about `54-61 ms`.
- Server-side prefill time dropped from about `5.0s` to about `0.46-0.51s`.

This is the expected PagedAttention/APC shape: prefix caching improves prefill
and TTFT. ITL stayed around `27-29 ms`, which is consistent with APC not being a
decode optimization.

## Decision

Keep the serving default at `block_size=16` and `prefix_caching_hash_algo=sha256`
for now.

Reasoning:

- `block_size=16` gives finer block granularity and matches the agent
  system/developer-prefix use case.
- `block_size=32` was not clearly faster; it also had one transient 4096-context
  startup failure at the KV boundary before the retry succeeded.
- `xxhash` did not show a meaningful benefit at concurrency `1`; keep it as a
  later high-QPS CPU-hash experiment rather than a default.
- `sha256` remains the safer default hash algorithm.

## Artifacts

```text
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block16-sha256.csv
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block16-sha256.jsonl
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block32-sha256.csv
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block32-sha256.jsonl
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block16-xxhash.csv
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block16-xxhash.jsonl
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block32-xxhash.csv
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block32-xxhash.jsonl
```
