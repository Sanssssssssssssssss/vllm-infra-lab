# vLLM Prefix Cache Block Experiment

Date: 2026-05-05

Profile:

- Backend: `vllm`
- Model: `Qwen3-8B-GGUF-vLLM-local`
- Quantization: `gguf-q4_k_m`
- Route: `WSL`
- vLLM: `0.17.1`
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- VRAM: `8188 MB`
- `max_model_len=4096`
- `max_num_seqs=2`
- `max_num_batched_tokens=4096`
- `gpu_memory_utilization=0.80`
- `block_size=16`
- Prefix caching: enabled
- Chunked prefill: enabled
- Async scheduling: enabled
- Enforce eager: false

Artifacts:

- `reports/benchmarks/2026-05-05-vllm-prefix-cache-blocks.csv`
- `reports/benchmarks/2026-05-05-vllm-prefix-cache-blocks.jsonl`
- Server log: `logs/vllm_stage3_prefix_blocks.log` (not committed)

## Result Summary

| case | prompt tokens | output tokens | shared blocks | TTFT p50 ms | E2E p50 ms | ITL p50 ms | prefix hit rate | prefill sum s | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case_a_distinct_1024` | 1022 | 26 | 0 | 4225.72 | 4898.66 | 24.72 | 0.0000 | 42.1419 | 0 |
| `case_b_shared_system_1024` | 1063 | 19 | 64 | 185.36 | 641.43 | 24.88 | 0.9632 | 1.7697 | 0 |
| `case_c_shared_document_2048` | 2089 | 16 | 128 | 145.66 | 483.77 | 25.95 | 0.9880 | 1.3237 | 0 |

## Interpretation

This run shows APC doing exactly what we wanted to prove:

- Case A had zero prefix-cache hits and paid full prefill cost.
- Case B reused the 1024-token system prefix, which corresponds to about 64
  full KV blocks at `block_size=16`.
- Case C reused the 2048-token document prefix, about 128 full KV blocks.
- TTFT collapsed from 4.2 seconds to roughly 0.15-0.19 seconds for shared-prefix
  cases.
- Server-side prefill time sum dropped from 42.14 seconds to 1.77 seconds and
  1.32 seconds.
- ITL stayed around 25 ms/token across all cases, which supports the key point:
  prefix caching reduces prefill work, not decode step cost.

The aggregate `output_tps` rose in the shared-prefix cases because this benchmark
uses short outputs and computes TPS over full wall-clock time. That metric
therefore still includes prefill delay. For a pure decode comparison, use ITL or
run a follow-up with long outputs such as 512 tokens.

## Log Evidence

The vLLM server log also reported increasing prefix-cache hit rate during the
shared-prefix rows, ending near 68.8 percent for the cumulative server lifetime.
The per-case CSV metrics are more useful because they subtract `/metrics`
counter baselines around each case.

## Next Experiment

Run the same cases with a long output target to make decode dominate:

```bash
python ./scripts/bench_prefix_cache_blocks.py \
  --cases case_a_distinct_1024,case_b_shared_system_1024 \
  --request-count 10 \
  --concurrency 1 \
  --override-output-tokens 512 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --notes stage3-prefix-cache-long-decode
```
