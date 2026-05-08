# Quantized Model Selection

This note records which local quantized models are intended for vLLM serving and why.

## Storage Layout

Keep model artifacts inside the repository workspace but out of git:

```text
models/
  Qwen3-8B/
  Qwen3-8B-GGUF/
  Qwen3-8B-AWQ/
  claude2-alpaca-7B-AWQ/
hf-cache/
```

Both `models/` and `hf-cache/` are ignored by git. This keeps C: clean, keeps
the experiment self-contained on E:, and avoids publishing multi-GB weights.

## Current Recommendation

Use `Qwen/Qwen3-8B-AWQ` through vLLM with AWQ-Marlin:

```bash
bash ./scripts/start_vllm_qwen3_awq_wsl.sh /mnt/e/GPTProject2/vLLM
```

Default profile:

```text
model = models/Qwen3-8B-AWQ
served_model_name = Qwen3-8B-AWQ-vLLM-local
quantization = awq_marlin
dtype = auto
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

The first attempt with plain `awq`, `gpu_memory_utilization=0.80`, and
`max_num_seqs=8` failed on the 8GB GPU because vLLM reported negative available
KV cache memory after model load. The working profile switches to `awq_marlin`,
raises the memory target moderately, caps sequence concurrency, and uses eager
execution to avoid CUDA graph reserve pressure.

## Claude-Distilled Candidate

`TheBloke/claude2-alpaca-7B-AWQ` was tested because it is a Claude-style
distilled/alpaca model available on Hugging Face in AWQ form. It works with vLLM
when using the local Alpaca chat template:

```bash
bash ./scripts/start_vllm_claude2_awq_wsl.sh /mnt/e/GPTProject2/vLLM
```

It is kept as a learning comparison target, not the active serving default. On
this GPU and vLLM build its decode latency was much worse than both Qwen GGUF
and Qwen AWQ-Marlin.

## Benchmark Artifacts

The first GGUF-vs-AWQ comparison is recorded in:

```text
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-gguf.csv
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-qwen3-awq-marlin.csv
reports/benchmarks/2026-05-08-vllm-gguf-vs-awq-claude2-awq.csv
reports/2026-05-08-vllm-gguf-vs-awq.md
```

The required next step is to rerun the full matrix at concurrency `1,2,4,8`.
This first pass only used `1,2` so the model switch could be validated quickly
before deeper tuning.
