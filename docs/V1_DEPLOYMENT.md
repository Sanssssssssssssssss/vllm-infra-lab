# V1 Deployment Plan

## Goal
Deploy `Qwen3-8B` on this laptop and expose an OpenAI-compatible API over the local network for another machine's agent to call.

## Confirmed Host Environment
- OS: Windows 11 Home 64-bit
- CPU: Intel Core Ultra 9 185H
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- VRAM observed by `nvidia-smi`: 8188 MiB
- System RAM: 32 GB
- CUDA runtime reported by driver: 12.7
- WSL status: `wsl.exe` exists, but no Linux distribution is installed yet

## Chosen V1 Route
Use `WSL2 + Ubuntu + Python venv + vLLM + OpenAI-compatible server`.

Why this route:
- vLLM officially ships pre-built CUDA wheels for Linux x86_64, not native Windows.
- WSL2 is the lowest-friction route on the current machine that still preserves CUDA access.
- The OpenAI-compatible server reduces integration work for the remote agent.
- The Python virtual environment should live on the Linux filesystem, not under `/mnt/e`, to avoid WSL venv permission issues.

## Assumptions Locked For V1
- Access scope is LAN only
- API scope is the minimum useful OpenAI-compatible subset
- One model profile only
- Focus is "working remote inference baseline" before AWQ, LoRA, distillation, or tracing

## API Scope For V1
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Why CPU Offload Is Enabled In The Baseline
`Qwen3-8B` is large relative to an 8 GB laptop GPU. The baseline keeps the original model target and uses CPU offload plus conservative runtime settings to maximize the chance of a first successful launch. The first stable baseline in this repository uses:
- `VLLM_GPU_MEMORY_UTILIZATION=0.80`
- `VLLM_MAX_NUM_SEQS=1`
- `VLLM_MAX_MODEL_LEN=2048`

This is a functionality-first baseline, not the final performance configuration.

## Recommended Sequence
1. Install WSL Ubuntu.
2. In WSL, bootstrap Python and vLLM.
3. In WSL, download the model to a local directory.
4. In WSL, copy `config/runtime.env.example` to `config/runtime.env` and adjust the API key.
5. In WSL, start the vLLM server.
6. In WSL, validate local API health.
7. From WSL, trigger Windows LAN exposure setup when needed.
8. Validate from the second computer over LAN.
9. Record baseline performance before moving to AWQ.

## Step 1: Install Ubuntu For WSL2
From an elevated PowerShell window:

```powershell
wsl --install -d Ubuntu
```

If the installer requests a reboot, reboot and finish the Ubuntu first-run setup.

## Step 2: Bootstrap The Linux Runtime
After Ubuntu is available, open the Ubuntu terminal and run:

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/bootstrap_vllm_wsl.sh /mnt/e/GPTProject2/vLLM
```

That script will:
- create `~/.venvs/gptproject2-vllm`
- upgrade `pip`, `setuptools`, and `wheel`
- install pinned Python packages for the first deployment baseline

## Step 3: Download The Model To A Local Directory

Run in WSL:

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/download_model_wsl.sh /mnt/e/GPTProject2/models/Qwen3-8B Qwen/Qwen3-8B
```

## Step 4: Prepare Runtime Config

```bash
cd /mnt/e/GPTProject2/vLLM
cp config/runtime.env.example config/runtime.env
```

Then edit:
- `VLLM_API_KEY`
- `HF_HOME` if you want a different cache location
- memory-related values if launch fails
- keep `HF_HUB_DISABLE_XET=1` if the default Xet-based download path stalls under WSL

Recommended first values on the current laptop:
- `VLLM_GPU_MEMORY_UTILIZATION=0.80`
- `VLLM_MAX_NUM_SEQS=1`
- `VLLM_MAX_MODEL_LEN=2048`
- `VLLM_ENFORCE_EAGER=1`
- `VLLM_SERVED_MODEL_NAME=Qwen3-8B-local`

## Step 5: Start The Service

Run in WSL:

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/start_vllm_wsl.sh /mnt/e/GPTProject2/vLLM
```

Default server address:
- Host bind: `0.0.0.0`
- Port: `8000`

## Step 6: Validate Locally

Run in WSL:

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/healthcheck_vllm_wsl.sh 127.0.0.1 8000 YOUR_KEY
bash ./scripts/chat_test_vllm_wsl.sh /mnt/e/GPTProject2/vLLM --host 127.0.0.1 --port 8000 --api-key YOUR_KEY --model /mnt/e/GPTProject2/models/Qwen3-8B --message "只回答：本机验证成功。" --max-tokens 12 --timeout 120 --disable-thinking
bash ./scripts/ask_vllm_wsl.sh /mnt/e/GPTProject2/vLLM "用一句中文介绍你自己。" 32 120 off
```

## Step 6.1: Ask From Another Laptop

If the second laptop also has this repository available in WSL, use:

```bash
cd /path/to/this/repo
bash ./scripts/ask_remote_vllm_wsl.sh 10.254.157.15 change-this-before-lan-use "只回答：第二台电脑提问成功。" Qwen3-8B-local 16 120 off
```

If you do not want to copy the repository to the second laptop, use plain `curl`:

```bash
curl http://10.254.157.15:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-this-before-lan-use" \
  -d '{
    "model": "Qwen3-8B-local",
    "messages": [{"role": "user", "content": "只回答：第二台电脑提问成功。"}],
    "max_tokens": 16,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Step 7: Trigger Windows LAN Exposure Setup From WSL

Run in WSL:

```bash
cd /mnt/e/GPTProject2/vLLM
bash ./scripts/configure_windows_lan_access_wsl.sh /mnt/e/GPTProject2/vLLM 8000
```

What this does:
- stays in the WSL workflow as the primary entrypoint
- opens an elevated Windows PowerShell window only for `portproxy` and firewall setup

Note:
- The WSL IP can change after restart, so the portproxy may need to be refreshed.
- This is the only step that still depends on Windows admin rights.

## Step 8: Validate From Another Computer
Use this machine's LAN IP, for example:
- `10.254.157.15:8000` on WLAN
- `10.7.0.2:8000` on Ethernet 3

Example:

```bash
curl http://10.254.157.15:8000/health
```

## Expected First Risks
- First launch may still need lower `VLLM_MAX_NUM_SEQS`
- `VLLM_CPU_OFFLOAD_GB` may need tuning
- Laptop background GPU usage reduces available VRAM
- Windows firewall may block inbound access until a rule is added
- WSL2 may require Windows portproxy configuration before another machine can reach the service
- Windows 任务管理器中的“共享 GPU 内存”不能等价视为可被 vLLM 当作高速显存主动吃满

## After The First Successful Launch
Priority order:
1. record latency and throughput baseline
2. reduce background GPU memory usage
3. evaluate AWQ as the first performance-oriented optimization path
4. optimize for lower offload and better tokens/s, not for higher shared-memory occupancy

## Reference Links
- [vLLM installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/index.html)
- [vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/serving/engine_args.html)
- [Microsoft WSL GPU compute docs](https://learn.microsoft.com/windows/wsl/tutorials/gpu-compute)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
