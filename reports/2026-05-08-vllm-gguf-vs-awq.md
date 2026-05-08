# 2026-05-08 vLLM GGUF vs AWQ Comparison

## Goal

Move model artifacts into the repository workspace, find a vLLM-friendlier
quantized model than the current GGUF path, and benchmark it against the GGUF
baseline before continuing optimization work.

## Model Storage

Model artifacts were moved under:

```text
E:\GPTProject2\vLLM\models\
```

The old outer folder `E:\GPTProject2\models` is now empty. `models/` and
`hf-cache/` are ignored by git so the public repository only stores scripts,
docs, and benchmark artifacts.

## Candidates

| candidate | source | quantization | result |
| --- | --- | --- | --- |
| Qwen3-8B GGUF | local existing model | GGUF Q4_K_M | Baseline kept working after repo-local move. |
| Claude2-Alpaca 7B AWQ | `TheBloke/claude2-alpaca-7B-AWQ` | AWQ int4 | Works with an Alpaca chat template, but decode is too slow here. |
| Qwen3-8B AWQ | `Qwen/Qwen3-8B-AWQ` | AWQ-Marlin int4 | Recommended active vLLM quantized model. |

The Claude-style candidate was included because the project preference was to
try a Claude-distilled Hugging Face model when possible. The benchmark result
does not support making it the active serving model on this 8GB RTX 4070 Laptop
GPU.

## Working Qwen3 AWQ-Marlin Profile

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
prefix_caching = true
chunked_prefill = true
async_scheduling = true
```

The first Qwen AWQ launch failed with `quantization=awq`,
`gpu_memory_utilization=0.80`, and `max_num_seqs=8`: vLLM loaded about 5.71 GiB
of model memory and then reported negative available KV cache memory. vLLM also
reported that the model could run with `awq_marlin`, which became the tested
working path.

## Concurrency 2 Summary

| workload | model | TTFT p50 ms | ITL p50 ms | E2E p50 ms | output TPS |
| --- | --- | ---: | ---: | ---: | ---: |
| short_chat | GGUF Q4_K_M | 151.50 | 32.85 | 4407.09 | 57.84 |
| short_chat | Qwen AWQ-Marlin | 70.51 | 25.18 | 3294.88 | 77.51 |
| short_chat | Claude2 AWQ | 246.62 | 162.61 | 26904.10 | 9.48 |
| long_prefill | GGUF Q4_K_M | 154.17 | 35.61 | 1287.00 | 47.16 |
| long_prefill | Qwen AWQ-Marlin | 59.00 | 26.19 | 864.05 | 68.78 |
| long_prefill | Claude2 AWQ | 518.96 | 194.15 | 5766.11 | 7.98 |
| long_decode | GGUF Q4_K_M | 157.48 | 33.32 | 17588.85 | 58.15 |
| long_decode | Qwen AWQ-Marlin | 82.01 | 25.59 | 13555.20 | 75.48 |
| long_decode | Claude2 AWQ | 171.40 | 157.88 | 92588.13 | 11.06 |
| shared_prefix | GGUF Q4_K_M | 152.11 | 33.88 | 1246.71 | 48.36 |
| shared_prefix | Qwen AWQ-Marlin | 65.42 | 25.65 | 856.11 | 70.64 |
| shared_prefix | Claude2 AWQ | 177.42 | 158.49 | 5705.03 | 11.22 |

## Decision

Use Qwen3-8B AWQ-Marlin as the active vLLM quantized model for the next tuning
round. It improves TTFT, ITL, E2E latency, and aggregate output TPS over the
GGUF baseline in this quick matrix, while staying within the 8GB VRAM envelope.

Keep Claude2-Alpaca AWQ available as a comparison model, but do not optimize
around it unless the learning goal shifts toward model behavior rather than
serving throughput.

## Artifacts

```text
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-gguf.csv
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-gguf.jsonl
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-qwen3-awq-marlin.csv
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-qwen3-awq-marlin.jsonl
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-claude2-awq.csv
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-claude2-awq.jsonl
```

## Notes

This was an intentionally small comparison matrix: four required workloads at
concurrency `1,2` with `waves=1`. The next clean benchmark should run the full
required concurrency matrix `1,2,4,8` against the selected Qwen AWQ-Marlin
profile.
