# vLLM Optimization Notes

## Current Working Optimized Route

Validated on 2026-05-05:

- Host: Windows 11 + WSL2 Ubuntu
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM
- vLLM: `0.17.1`
- Model: `/mnt/e/GPTProject2/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`
- Tokenizer/config source: `/mnt/e/GPTProject2/models/Qwen3-8B`
- Served model name: `Qwen3-8B-GGUF-vLLM-local`
- Endpoint: `http://127.0.0.1:8000`

Start in WSL:

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/start_vllm_gguf_optimized_wsl.sh /mnt/e/GPTProject2/vLLM
```

Ask in WSL:

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/ask_vllm_gguf_optimized_wsl.sh /mnt/e/GPTProject2/vLLM "只回答：vLLM 本地聊天服务已经启动成功。" 48 180 off
```

## Enabled Optimization Knobs

The current optimized route intentionally exercises vLLM infra features rather than only serving a single request:

- `--enable-prefix-caching` for automatic prefix caching.
- `--enable-chunked-prefill` with `--max-num-batched-tokens 4096` for prefill scheduling.
- `--max-num-seqs 8` so continuous batching has headroom.
- `--async-scheduling` to reduce GPU idle gaps from scheduler overhead.
- `--block-size 16` for KV cache block granularity.
- CUDA graph is left enabled by not passing `--enforce-eager`.
- `--generation-config vllm` avoids model generation defaults unexpectedly constraining requests.

## Observed Startup Facts

The first validated optimized launch produced:

- GGUF model load: about `4.82 GiB` GPU memory.
- Available KV cache memory: about `1.13 GiB`.
- GPU KV cache size: `8,224 tokens`.
- Maximum concurrency for `2,048` tokens per request: about `4.02x`.
- FlashAttention selected automatically.
- CUDA graph capture completed successfully.
- A 4-request local concurrency smoke test completed successfully, confirming the endpoint can accept concurrent chat requests. Treat this as a smoke test, not a benchmark.

The first attempt with `--gpu-memory-utilization 0.86` failed because free startup VRAM was just under the requested allocation. The working value is currently `0.80`.

## Next Optimization Matrix

For the next benchmarking round, vary one axis at a time:

- Prefill: `VLLM_MAX_NUM_BATCHED_TOKENS=2048/4096/8192`, with chunked prefill on/off.
- Continuous batching: `VLLM_MAX_NUM_SEQS=1/4/8`, using concurrent requests.
- KV cache capacity: `VLLM_MAX_MODEL_LEN=1024/2048/4096`, `VLLM_KV_CACHE_DTYPE=fp8_e5m2` if quality is acceptable.
- Prefix cache: repeat shared system/developer prefixes and compare first request versus warm-prefix requests.
- CUDA graph: compare default graph mode against `VLLM_ENFORCE_EAGER=1` as a control.

Metrics to record later:

- Time to first token.
- Output tokens per second.
- End-to-end latency.
- Prompt tokens, completion tokens, and concurrency.
- GPU memory used, KV cache tokens, cache hit behavior, and error/OOM boundary.
