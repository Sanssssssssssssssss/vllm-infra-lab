# 2026-05-10 AWQ-Marlin Speculative Decoding Pass

## Scope

Goal: test whether n-gram / suffix speculative decoding reduces target-model
sequential decode work for the active AWQ-Marlin route.

Base profile:

```text
model = Qwen3-8B-AWQ-vLLM-local
quantization = awq-marlin-int4
vllm_version = 0.17.1
route = WSL
max_model_len = 2048
max_num_seqs = 2
max_num_batched_tokens = 4096
gpu_memory_utilization = 0.85
block_size = 16
prefix_caching = on
chunked_prefill = on
enforce_eager = false
cuda_graph = true
```

Benchmark scope:

```text
workload = long_decode
prompt target = 128 tokens
output target = 512 tokens
concurrency = 1, 2, 4
waves = 3
streaming = true
```

## Compatibility Boundaries

- `baseline` kept the active profile with async scheduling enabled.
- `baseline_sync` disabled async scheduling so n-gram could be compared fairly.
- vLLM `0.17.1` rejects async scheduling with n-gram speculative decoding:
  `Currently, async scheduling is only supported with EAGLE/MTP/Draft Model
  kind of speculative decoding.`
- Suffix decoding did not start because `arctic-inference==0.1.1` is missing.
  A direct install attempt was stopped after it spent more than 10 minutes
  building/downloading a separate dependency stack, including `torch==2.7.0`.

## Main Result

N-gram speculation massively improved aggregate output TPS and E2E latency on
this synthetic long-decode workload, but it did not improve client-observed ITL.
The benchmark prompt asks the model to produce repetitive benchmark text, so
this is a best-case n-gram result rather than a general chat result.

| profile | c | ITL p50 ms | ITL p95 ms | E2E p95 ms | output TPS | error_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_sync | 1 | 23.57 | 25.96 | 12722.36 | 41.27 | 0 |
| ngram4_sync | 1 | 26.54 | 28.60 | 4364.67 | 151.82 | 0 |
| ngram8_sync | 1 | 26.96 | 29.78 | 2643.98 | 256.97 | 0 |
| baseline_sync | 2 | 23.98 | 26.15 | 12430.78 | 82.44 | 0 |
| ngram4_sync | 2 | 27.74 | 30.12 | 2962.29 | 347.72 | 0 |
| ngram8_sync | 2 | 28.84 | 30.98 | 1766.88 | 597.65 | 0 |
| baseline_sync | 4 | 24.05 | 26.20 | 24900.37 | 82.37 | 0 |
| ngram4_sync | 4 | 27.92 | 30.46 | 5923.27 | 348.55 | 0 |
| ngram8_sync | 4 | 29.47 | 31.83 | 3454.46 | 592.74 | 0 |

Active-profile control:

| profile | c | ITL p50 ms | ITL p95 ms | E2E p95 ms | output TPS | error_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline async-on | 1 | 23.65 | 25.20 | 12469.08 | 41.66 | 0 |
| baseline async-on | 2 | 24.10 | 25.69 | 12436.89 | 82.42 | 0 |
| baseline async-on | 4 | 24.13 | 25.81 | 24857.32 | 82.40 | 0 |

## Speculative Counters

| variant | draft tokens | accepted tokens | acceptance rate | mean acceptance length |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 0 | n/a | n/a |
| baseline_sync | 0 | 0 | n/a | n/a |
| ngram4_sync | 8652 | 8652 | 1.000000 | 5.000000 |
| ngram8_sync | 9576 | 9576 | 1.000000 | 9.000000 |

Per-position acceptance was `1.0` at every drafted position for both n-gram
variants. This confirms n-gram speculation was active and perfectly accepted
on the benchmark's repetitive output shape.

## Decision

Do not switch the default agent/chat profile to n-gram speculative decoding.

Reasons:

- The user's hard gate was long_decode ITL p50/p95. ITL did not drop; it
  increased from `baseline_sync` p95 `25.96-26.20 ms` to `29.78-31.83 ms` for
  `ngram8_sync`.
- n-gram requires async scheduling off in vLLM `0.17.1`, so it is not a
  drop-in upgrade for the current active profile.
- The 100% acceptance rate is caused by the synthetic repeated-text workload.
  It proves the mechanism works, but not that normal chat/tool-use traffic will
  benefit.

Keep `ngram8_sync` as an experimental repetitive-output/copy-like workload
profile. It reduced E2E p95 by about `79-86%` and raised output TPS from about
`82 tok/s` to about `593-598 tok/s` at c2/c4, with `error_count=0`.

## Artifacts

Benchmarks:

- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-baseline-waves3.csv`
- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-baseline_sync-waves3.csv`
- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-ngram4_sync-waves3.csv`
- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-ngram8_sync-waves3.csv`
- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-metrics.csv`

Startup boundaries:

- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-ngram4-waves3-startup-failure.txt`
- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-ngram8-waves3-startup-failure.txt`
- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-suffix-waves3-startup-failure.txt`
- `reports/benchmarks/2026-05-10-vllm-awq-marlin-specdecode-suffix_sync-waves3-startup-failure.txt`

Raw metric snapshots:

- `reports/metrics/2026-05-10-vllm-awq-marlin-specdecode-*-before.prom`
- `reports/metrics/2026-05-10-vllm-awq-marlin-specdecode-*-after.prom`
