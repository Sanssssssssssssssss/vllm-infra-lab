# 2026-05-10 AWQ-Marlin Performance Summary

## Active Serving Profile

This is the current default profile for agent/chat serving:

```text
backend = vllm
model = Qwen3-8B-AWQ-vLLM-local
quantization = awq-marlin-int4
route = WSL
gpu = NVIDIA GeForce RTX 4070 Laptop GPU, 8GB VRAM
vllm_version = 0.17.1
max_model_len = 2048
max_num_seqs = 2
max_num_batched_tokens = 4096
gpu_memory_utilization = 0.85
block_size = 16
kv_cache_dtype = auto
prefix_caching = on
chunked_prefill = on
async_scheduling = on
enforce_eager = false
cuda_graph = true
streaming = true
```

Startup snapshot:

| model GiB | KV GiB | GPU KV tokens | max concurrency estimate | GPU used after start |
| ---: | ---: | ---: | ---: | ---: |
| 5.71 | 0.49 | 3600 | 1.76x at 2048 tokens/request | 7332 MB |

## Request Benchmark

Source artifact:

```text
reports/benchmarks/2026-05-09-vllm-awq-marlin-eager-graph-b-graph-util085-seq2-waves3.csv
```

Workloads:

| workload | prompt target | output target | purpose |
| --- | ---: | ---: | --- |
| short_chat | 128 | 128 | interactive chat |
| long_prefill | 1024 | 32 | prefill / TTFT pressure |
| long_decode | 128 | 512 | decode / ITL pressure |
| shared_prefix | 1024 shared-prefix | 32 | APC-style prompt reuse |

Each row uses `waves=3`, concurrency `1,2,4`, and `error_count=0`.

## Headline Numbers

| scenario | c1 output TPS | c2 output TPS | c4 output TPS | c1 TTFT p95 | c2 TTFT p95 | c4 TTFT p95 | ITL p95 range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_chat | 34.76 | 71.44 | 70.91 | 295.22 ms | 71.81 ms | 3682.26 ms | 30.02-33.19 ms |
| long_prefill | 28.97 | 61.63 | 65.17 | 443.74 ms | 92.24 ms | 1023.10 ms | 30.45-32.25 ms |
| long_decode | 35.91 | 70.45 | 70.46 | 49.45 ms | 80.44 ms | 14640.23 ms | 30.28-30.84 ms |
| shared_prefix | 28.20 | 61.41 | 64.85 | 506.46 ms | 100.40 ms | 1026.64 ms | 30.61-32.09 ms |

Interpretation:

- Output throughput is roughly `35 output tok/s` at c1 and `70 output tok/s`
  at c2/c4 for decode-heavy traffic.
- `max_num_seqs=2` is visible in the data: c2 is the sweet spot; c4 does not
  double output TPS and mainly adds queueing.
- Decode smoothness is good for this laptop GPU: ITL p95 stays around
  `30-33 ms/token` in the current profile.
- Short chat c4 TTFT p95 is already several seconds, so c4 is acceptable for
  batch/throughput experiments but not ideal for latency-sensitive agent UX.

## Detailed Table

| workload | c | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | E2E p95 ms | output TPS | total TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_chat | 1 | 46.78 | 295.22 | 26.90 | 33.19 | 4015.02 | 34.76 | 69.52 |
| short_chat | 2 | 67.64 | 71.81 | 27.30 | 30.02 | 3583.70 | 71.44 | 142.88 |
| short_chat | 4 | 3672.09 | 3682.26 | 27.47 | 30.39 | 7240.13 | 70.91 | 141.81 |
| long_prefill | 1 | 48.93 | 443.74 | 27.97 | 30.45 | 1353.44 | 28.97 | 956.15 |
| long_prefill | 2 | 83.53 | 92.24 | 28.79 | 32.25 | 1035.96 | 61.63 | 2033.93 |
| long_prefill | 4 | 983.47 | 1023.10 | 28.58 | 31.14 | 1939.41 | 65.17 | 2150.66 |
| long_decode | 1 | 45.76 | 49.45 | 27.39 | 30.28 | 14284.29 | 35.91 | 44.88 |
| long_decode | 2 | 77.05 | 80.44 | 27.90 | 30.84 | 14555.96 | 70.45 | 88.06 |
| long_decode | 4 | 14576.47 | 14640.23 | 27.90 | 30.84 | 29101.33 | 70.46 | 88.07 |
| shared_prefix | 1 | 46.39 | 506.46 | 28.14 | 30.61 | 1424.63 | 28.20 | 930.69 |
| shared_prefix | 2 | 84.49 | 100.40 | 28.91 | 31.80 | 1040.14 | 61.41 | 2026.39 |
| shared_prefix | 4 | 990.23 | 1026.64 | 28.70 | 32.09 | 1949.82 | 64.85 | 2140.16 |

## Prefix Cache Check

Source artifact:

```text
reports/benchmarks/2026-05-09-vllm-awq-marlin-prefix-cache-block16-sha256.csv
```

| case | shared blocks | prompt tokens | TTFT p50 ms | TTFT p95 ms | prefix hits | hit rate | prefill sum s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distinct 1024 | 0 | 1022 | 486.20 | 664.79 | 0 | 0.0000 | 5.0183 |
| shared system 1024 | 64 | 1063 | 54.50 | 64.48 | 10240 | 0.9632 | 0.4665 |
| shared document 2048 | 128 | 2089 | 58.50 | 65.60 | 20640 | 0.9880 | 0.4618 |

APC is confirmed active. Shared-prefix TTFT is roughly `8-9x` lower than the
distinct-prompt control, and server-side prefill time drops from about `5s` to
about `0.46s`.

## Long-Context Experimental Profile

For 4096-token context, the experimental FP8 KV profile is:

```text
kv_cache_dtype = fp8_e4m3
attention_backend = TRITON_ATTN
max_model_len = 4096
max_num_seqs = 2
max_num_batched_tokens = 2048
enforce_eager = true
```

Capacity snapshot:

| profile | status | GPU KV tokens | max concurrency estimate | quality regression |
| --- | --- | ---: | ---: | --- |
| auto + TRITON_ATTN | ready | 4704 | 1.15x | 5/5 pass |
| fp8_e4m3 + TRITON_ATTN | ready | 8512 | 2.08x | 5/5 pass |

This profile is not the default because it requires eager mode and a backend
change. Use it for long-context/KV-capacity experiments, not for the normal
short-chat agent backend.

## Practical Recommendation

- Best default for agent/chat: current AWQ-Marlin CUDA graph profile.
- Best latency target: keep practical concurrency at `1-2`.
- Best throughput target on this GPU: c2 gives about `70 output tok/s`; c4 mostly
  queues because `max_num_seqs=2`.
- Watch metric for UX: `short_chat TTFT p95`. It is good at c1/c2, bad at c4.
- Watch metric for decode smoothness: `ITL p95`, currently about `30-33 ms`.
