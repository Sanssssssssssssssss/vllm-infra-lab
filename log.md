# LOG

## 目的
记录本项目在 Linux / WSL Ubuntu 中已经验证过的可复现 CLI。

记录规则：
- 只记录最终可跑通的命令
- 不记录试错命令
- 后续只要出现新的“已验证成功”的 Linux CLI，就追加到本文件

日常操作约定：
- 优先在 WSL Ubuntu 内完成模型、环境、服务相关操作
- 只有 Windows 网络暴露相关操作（`portproxy`、防火墙）保留在管理员 PowerShell
- 但从用户视角，默认仍从 WSL 发起，由 WSL 脚本拉起 Windows 提权窗口

## 当前可跑通路径

### 1. 安装 Python 运行组件
在 Ubuntu 中补齐 `venv` 和 `pip`：

```bash
apt-get update && apt-get install -y python3-venv python3-pip
```

说明：
- 这是在 WSL Ubuntu 内执行的 root 命令
- 用于解决新装 Ubuntu 缺少完整 `venv/pip` 组件的问题

### 2. 创建项目虚拟环境
将虚拟环境放在 Linux 文件系统中，而不是 `/mnt/e`：

```bash
python3 -m venv ~/.venvs/gptproject2-vllm
source ~/.venvs/gptproject2-vllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install vllm==0.17.1 openai
```

说明：
- 虚拟环境放在 `~/.venvs/gptproject2-vllm`
- 这是为了避免 WSL 在 Windows 挂载盘上创建 venv 时的权限和 `ensurepip` 问题

### 3. 校验 GPU 在 WSL 内可用

```bash
source ~/.venvs/gptproject2-vllm/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
```

成功判据：
- `torch.cuda.is_available()` 为 `True`
- 能打印 `NVIDIA GeForce RTX 4070 Laptop GPU`

### 4. 下载模型到本地目录
先下载模型，再启动服务：

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/download_model_wsl.sh /mnt/e/GPTProject2/models/Qwen3-8B Qwen/Qwen3-8B
```

说明：
- `HF_HUB_DISABLE_XET=1` 是当前机器上已验证更稳定的下载方式
- 下载完成后，服务直接从本地目录加载模型

### 5. 启动 vLLM 服务
进入项目目录并加载本地模型：

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/start_vllm_wsl.sh /mnt/e/GPTProject2/vLLM
```

说明：
- `--enforce-eager` 是当前机器上已验证可跑通的关键参数
- 不加它时，`CPU offload + torch.compile` 会触发 `torch._dynamo` 失败

### 6. 查看服务是否监听

```bash
ss -ltnp | grep 8000
```

成功判据：
- 出现 `0.0.0.0:8000`

### 7. 在 Linux 内做健康检查

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/healthcheck_vllm_wsl.sh 127.0.0.1 8000 YOUR_KEY
```

### 8. 在 Linux 内做聊天验证

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/chat_test_vllm_wsl.sh /mnt/e/GPTProject2/vLLM --host 127.0.0.1 --port 8000 --api-key YOUR_KEY --model /mnt/e/GPTProject2/models/Qwen3-8B --message "只回答：当前服务仍在运行。" --max-tokens 12 --timeout 120 --disable-thinking
```

### 9. 更短的日常提问命令

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/ask_vllm_wsl.sh /mnt/e/GPTProject2/vLLM "只回答：当前服务仍在运行。" 12 120 off
```

说明：
- 默认读取 `config/runtime.env`
- 默认打本机 `127.0.0.1`
- 默认关闭 thinking，更适合 agent/脚本场景

### 10. 第二台笔记本远程 ask

```bash
cd /path/to/this/repo
bash ./scripts/ask_remote_vllm_wsl.sh 10.254.157.15 change-this-before-lan-use "只回答：第二台电脑提问成功。" Qwen3-8B-local 16 120 off
```

### 11. 从 WSL 触发 Windows 局域网暴露

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/configure_windows_lan_access_wsl.sh /mnt/e/GPTProject2/vLLM 8000
```

### 12. 当前首版运行参数摘要

```bash
export VLLM_MODEL=/mnt/e/GPTProject2/models/Qwen3-8B
export VLLM_SERVED_MODEL_NAME=Qwen3-8B-local
export VLLM_DTYPE=half
export VLLM_MAX_MODEL_LEN=2048
export VLLM_HOST=0.0.0.0
export VLLM_PORT=8000
export VLLM_API_KEY=change-this-before-lan-use
export VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_MAX_NUM_SEQS=1
export VLLM_CPU_OFFLOAD_GB=10
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_ENFORCE_EAGER=1
export HF_HOME=/mnt/e/GPTProject2/hf-cache
export HF_HUB_DISABLE_XET=1
```

## 当前已验证结论
- `Qwen3-8B` 可以在当前机器上通过 `vLLM + WSL2 + CPU offload + enforce-eager` 跑起来
- 本机 `localhost:8000` 上的 OpenAI 风格 API 已验证成功
- 第二台电脑可通过固定模型名 `Qwen3-8B-local` 发起请求
- 当前配置下生成速度较慢，约 `0.6-0.7 tokens/s`
- 局域网访问仍需 Windows 管理员权限配置 `portproxy` 和防火墙
- 当前 `cpu_offload_gb` 已在使用系统 RAM，但这不是把共享 GPU 内存变成高速显存，也通常不会提高 tokens/s

## 2026-05-05 vLLM GGUF 优化路线

### 13. 启动 vLLM GGUF 优化服务

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/start_vllm_gguf_optimized_wsl.sh /mnt/e/GPTProject2/vLLM
```

等价关键参数：

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

成功判据：
- `ss -ltnp | grep 8000` 出现 `0.0.0.0:8000`
- `GET /health` 返回成功
- 日志中出现 `FlashAttention`、`enable_prefix_caching=True`、`enable_chunked_prefill=True`、`Graph capturing finished`

### 14. 询问 vLLM GGUF 优化服务

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/ask_vllm_gguf_optimized_wsl.sh /mnt/e/GPTProject2/vLLM "只回答一句中文：vLLM 本地聊天服务已经启动成功。" 48 180 off
```

当前已验证返回：

```text
vLLM 本地聊天服务已经启动成功。
```

### 15. 当前观测

- `--gpu-memory-utilization 0.86` 启动失败，原因是 WSL 启动时可用显存略低于请求值
- `--gpu-memory-utilization 0.80` 启动成功
- GGUF 模型加载占用约 `4.82 GiB`
- 可用 KV cache memory 约 `1.13 GiB`
- GPU KV cache size 为 `8,224 tokens`
- `2048` tokens/request 下最大并发约 `4.02x`
- 4 个本机并发短请求已验证均返回成功；该结果只作为 continuous batching smoke test，不作为正式 benchmark

### 16. Stage 2 显存 / KV cache profiling pilot

```bash
cd /mnt/e/GPTProject2/vLLM
source ~/.venvs/gptproject2-vllm/bin/activate
python ./scripts/profile_vllm_memory_sweep.py --preset pilot --kill-existing --notes stage2-memory-pilot
```

说明：
- 该命令会重启 vLLM 服务
- 启动级 profiling 结果写入 `reports/memory/*.csv` 和 `reports/memory/*.jsonl`
- 原始启动日志写入 `logs/memory_profile/`，该目录不提交到 git
