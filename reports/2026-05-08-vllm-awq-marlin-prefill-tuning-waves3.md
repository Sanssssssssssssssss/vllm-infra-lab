# 2026-05-08 AWQ-Marlin Prefill Tuning

## Goal

Tune long-prompt / prefill pressure for the current AWQ-Marlin route and check
whether smaller scheduler token budgets prevent long prefill from hurting decode
or short-chat latency.

This pass used the retained interactive startup shape from the memory sweep:

```text
model = Qwen3-8B-AWQ-vLLM-local
quantization = awq_marlin
max_model_len = 2048
gpu_memory_utilization = 0.85
max_num_seqs = 2
enforce_eager = true
concurrency = 1,2,4,8
waves = 3
```

## Profiles

| group | max_num_batched_tokens | chunked prefill | status |
| --- | ---: | --- | --- |
| A | 2048 | on | benchmark complete |
| B | 4096 | on | benchmark complete |
| C | 8192 | on | startup failed |
| D | 2048 | off | benchmark complete |
| E | 4096 | off | benchmark complete |

Group C failed at startup: vLLM reported only `0.07 GiB` available KV cache
memory, while `0.28 GiB` was required to serve at least one `2048`-token request.
The estimated maximum model length was `464`. This makes `8192` an invalid
candidate for the current 8GB AWQ-Marlin route.

## Key c8 Results

| group | chunked | batch | short TTFT p95 | long_prefill TTFT p95 | long_prefill E2E p95 | shared_prefix TTFT p95 | long_decode ITL p95 | output TPS range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | on | 2048 | 12521.14 | 3259.44 | 4267.38 | 3305.05 | 41.50 | 59.03-63.05 |
| B | on | 4096 | 12658.34 | 3205.28 | 4208.66 | 3190.02 | 36.24 | 60.34-64.25 |
| D | off | 2048 | 12385.25 | 3237.32 | 4212.85 | 3201.53 | 38.92 | 60.47-63.89 |
| E | off | 4096 | 12615.69 | 3398.68 | 4426.44 | 3913.15 | 40.79 | 53.08-63.36 |

All benchmarked rows completed with `error_count=0`.

## Decision

Keep the serving default at:

```text
max_num_batched_tokens = 4096
chunked_prefill = on
```

Rationale:

- `2048` did not significantly lower `long_prefill` TTFT p95 versus `4096`.
- `4096/chunked=on` had the best c8 `long_prefill` TTFT p95, best c8
  `shared_prefix` TTFT p95, and best c8 `long_decode` ITL p95 among valid
  benchmarked profiles.
- `8192/chunked=on` failed startup and is not a valid candidate.
- Chunked prefill off was competitive in some rows, but it did not show a clear
  cross-workload advantage. Keep chunked prefill enabled.

## Caveat

This was a tuning pass with `waves=3`. It is enough to reject `8192`, reject the
off control as the default, and keep `4096/chunked=on`. Before calling this a
new production-like baseline, rerun the selected profile with `waves=5`.

## Artifacts

```text
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-a-batch2048-chunked-on-waves3.csv
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-a-batch2048-chunked-on-waves3.jsonl
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-b-batch4096-chunked-on-waves3.csv
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-b-batch4096-chunked-on-waves3.jsonl
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-c-batch8192-chunked-on-waves3-startup-failure.txt
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-d-batch2048-chunked-off-waves3.csv
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-d-batch2048-chunked-off-waves3.jsonl
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-e-batch4096-chunked-off-waves3.csv
reports/benchmarks/2026-05-08-vllm-awq-marlin-prefill-e-batch4096-chunked-off-waves3.jsonl
```
