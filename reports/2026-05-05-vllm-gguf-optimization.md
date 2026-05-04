# 2026-05-05 vLLM GGUF 优化启动报告

## 目标

本轮目标是把本地 `vLLM + WSL2` 推理服务先拉起来，形成可以聊天的 OpenAI-compatible API，并为后续学习 AI infra 中的 prefill、KV cache、continuous batching、automatic prefix caching 等主题保留可复现实验入口。

## 环境

- Host: Windows 11 + WSL2 Ubuntu
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM
- vLLM: `0.17.1`
- Model: `/mnt/e/GPTProject2/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`
- Tokenizer/config: `/mnt/e/GPTProject2/models/Qwen3-8B`
- Served model name: `Qwen3-8B-GGUF-vLLM-local`
- Endpoint: `http://127.0.0.1:8000`

## 当前启动参数

```bash
vllm serve /mnt/e/GPTProject2/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
  --tokenizer /mnt/e/GPTProject2/models/Qwen3-8B \
  --hf-config-path /mnt/e/GPTProject2/models/Qwen3-8B \
  --served-model-name Qwen3-8B-GGUF-vLLM-local \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key change-this-before-lan-use \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --block-size 16 \
  --generation-config vllm
```

## 已验证结论

- `GET /health` 成功。
- `GET /v1/models` 成功。
- `POST /v1/chat/completions` 成功。
- 4 个本机并发短请求 smoke test 均返回成功。
- FlashAttention 被自动选中。
- CUDA graph capture 成功完成。
- Automatic prefix caching、chunked prefill、async scheduling 均已启用。

## 启动观测

- `gpu_memory_utilization=0.86` 启动失败，原因是 WSL 启动时可用显存略低于请求值。
- `gpu_memory_utilization=0.80` 启动成功。
- GGUF 模型加载占用约 `4.82 GiB` GPU memory。
- Available KV cache memory 约 `1.13 GiB`。
- GPU KV cache size 为 `8,224 tokens`。
- `2048` tokens/request 下最大并发约 `4.02x`。
- 运行时 GPU 显存占用约 `7.5 GiB / 8 GiB`。

## 重要取舍

本轮没有继续使用之前的 fp16 safetensors + CPU offload baseline 作为优化起点，因为该路线需要 `--enforce-eager` 兜底，且单请求生成速度很慢。使用本地 GGUF 让模型权重更适配 8 GB laptop GPU，可以保留 vLLM 路线，并开始观察 scheduler、KV cache 和 CUDA graph 等 infra 特性。

## 后续实验矩阵

- Prefill: 对比 `max_num_batched_tokens=2048/4096/8192`，并开关 chunked prefill。
- Continuous batching: 对比 `max_num_seqs=1/4/8`，并使用并发请求。
- KV cache: 对比 `max_model_len=1024/2048/4096`，必要时评估 fp8 KV cache。
- Prefix caching: 固定长 system/developer 前缀，对比 cold prefix 与 warm prefix。
- CUDA graph: 对比默认 CUDA graph 与 `VLLM_ENFORCE_EAGER=1` 控制组。

## 建议记录指标

- Time to first token
- Output tokens/s
- End-to-end latency
- Prompt tokens / completion tokens
- Concurrency
- GPU memory used
- KV cache capacity and usage
- Prefix cache hit rate
- OOM or launch failure boundary
