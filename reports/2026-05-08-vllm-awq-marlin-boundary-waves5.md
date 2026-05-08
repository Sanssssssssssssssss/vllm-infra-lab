# 2026-05-08 Qwen3 AWQ-Marlin Boundary Baseline

## Goal

Find the real serving boundary for the current vLLM profile, especially whether
concurrency `4` and `8` cause timeout, OOM, queueing collapse, or p95 latency
regression.

Warmup was run first and discarded from the final artifact set:

```text
notes = awq-marlin-warmup-discard
workloads = short_chat,long_prefill,long_decode,shared_prefix
concurrency = 1,2,4,8
waves = 1
```

The formal baseline used `waves=5`:

```text
profile = qwen3_8b_awq_marlin_eager_vllm
model = Qwen3-8B-AWQ-vLLM-local
quantization = awq-marlin-int4
max_model_len = 2048
max_num_seqs = 2
max_num_batched_tokens = 4096
gpu_memory_utilization = 0.85
block_size = 16
enforce_eager = true
streaming = true
request_rate = 0
```

## Acceptance Summary

| check | result | notes |
| --- | --- | --- |
| `error_count` | pass | All 16 matrix rows had `error_count=0`. |
| `concurrency=4/8` timeout/OOM | pass | No request timeouts or OOM failures were recorded. |
| `long_decode ITL p95` | pass | c2 `49.52 ms`, c4 `49.49 ms`, c8 `39.59 ms`; decode did not explode. |
| `short_chat TTFT p95` | warning | c4 `4424.75 ms`, c8 `13229.48 ms`; agent UX degrades from queueing. |
| `gpu_mem_used_mb` | warning | Peak recorded value was `7516 MB` of `8188 MB`; stable, but low headroom. |

This profile is stable through concurrency `8`, but the usable interactive
boundary is much closer to concurrency `2`. At concurrency `4` and `8`, aggregate
throughput stays near saturation while TTFT absorbs the queue.

## Startup Memory

vLLM startup log:

```text
Model loading took 5.71 GiB memory
Available KV cache memory: 0.45 GiB
GPU KV cache size: 3,248 tokens
Maximum concurrency for 2,048 tokens per request: 1.59x
```

This explains the shape of the result: the model fits, but KV capacity and
`max_num_seqs=2` keep the engine in a small serving envelope. Bursts above c2
are accepted, but they mostly wait.

## Formal Baseline Results

| workload | c | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | E2E p95 ms | output TPS | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_chat | 1 | 57.15 | 315.59 | 28.99 | 37.22 | 4261.49 | 32.75 | 0 |
| short_chat | 2 | 81.41 | 100.41 | 29.38 | 36.94 | 4078.58 | 65.63 | 0 |
| short_chat | 4 | 4272.92 | 4424.75 | 32.78 | 40.14 | 8593.62 | 60.69 | 0 |
| short_chat | 8 | 12354.59 | 13229.48 | 30.84 | 39.29 | 17485.86 | 62.48 | 0 |
| long_prefill | 1 | 64.98 | 410.51 | 31.22 | 39.36 | 1352.95 | 27.73 | 0 |
| long_prefill | 2 | 102.39 | 126.26 | 35.62 | 41.94 | 1227.43 | 52.60 | 0 |
| long_prefill | 4 | 1094.76 | 1169.45 | 32.82 | 41.11 | 2185.00 | 58.26 | 0 |
| long_prefill | 8 | 3154.75 | 3432.16 | 31.47 | 39.94 | 4444.12 | 59.31 | 0 |
| long_decode | 1 | 71.67 | 78.91 | 31.45 | 38.04 | 17436.30 | 31.32 | 0 |
| long_decode | 2 | 90.08 | 111.25 | 35.25 | 49.52 | 20966.94 | 54.42 | 0 |
| long_decode | 4 | 18884.53 | 19970.12 | 34.31 | 49.49 | 39571.46 | 55.52 | 0 |
| long_decode | 8 | 46534.00 | 49394.21 | 29.33 | 39.59 | 65666.91 | 65.40 | 0 |
| shared_prefix | 1 | 80.37 | 433.49 | 32.20 | 39.69 | 1330.56 | 27.18 | 0 |
| shared_prefix | 2 | 90.47 | 115.72 | 34.99 | 40.18 | 1194.05 | 53.19 | 0 |
| shared_prefix | 4 | 1151.19 | 1187.71 | 34.42 | 39.33 | 2248.66 | 56.78 | 0 |
| shared_prefix | 8 | 2945.60 | 3374.15 | 29.98 | 38.90 | 4424.19 | 62.44 | 0 |

## Interpretation

The current profile is not failing at c4/c8; it is queueing. vLLM logs showed
`Running: 2 reqs, Waiting: 6 reqs` during the c8 burst, matching
`max_num_seqs=2`.

Decode quality is healthy: `long_decode ITL p95` at c4 is essentially identical
to c2, and c8 is lower in this run. Throughput also saturates around
`55-65 output tokens/s`, so adding more burst concurrency mostly moves work into
the waiting queue.

For an agent-facing local API, use concurrency `2` as the current practical
baseline. Treat concurrency `4` and `8` as stress/batch throughput modes until a
new profile reduces TTFT p95.

## Artifacts

```text
reports/benchmarks/2026-05-08-vllm-awq-marlin-boundary-waves5.csv
reports/benchmarks/2026-05-08-vllm-awq-marlin-boundary-waves5.jsonl
```
