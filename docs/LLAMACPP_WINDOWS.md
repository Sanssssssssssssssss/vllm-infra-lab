# llama.cpp Windows Route

## Goal
Use a Windows-friendly `llama.cpp + CUDA` backend for a quantized Qwen profile while keeping the existing `Qwen3-8B + vLLM` path available.

## Current Assumption
- Primary fast-path profile: `qwen25_7b_instruct_gguf_q4_k_m`
- Model repo: `Qwen/Qwen2.5-7B-Instruct-GGUF`
- Quantization: `q4_k_m`
- Backend: `llama.cpp`
- Host runtime: Windows 11 native

If you later want a different GGUF or another backend, add a new file under `config/profiles/` and switch with `scripts/set_active_profile.ps1`.

## Profile Commands

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\list_profiles.ps1 -WorkspaceDir E:\GPTProject2\vLLM
powershell -ExecutionPolicy Bypass -File .\scripts\set_active_profile.ps1 -WorkspaceDir E:\GPTProject2\vLLM -Profile qwen25_7b_instruct_gguf_q4_k_m
```

Switch back to the WSL + vLLM baseline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\set_active_profile.ps1 -WorkspaceDir E:\GPTProject2\vLLM -Profile qwen3_8b_vllm
```

## Bootstrap llama.cpp CUDA

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_llamacpp_windows.ps1 -WorkspaceDir E:\GPTProject2\vLLM
```

## Download The Active Model

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_active_model_windows.ps1 -WorkspaceDir E:\GPTProject2\vLLM
```

## Start And Validate

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_llamacpp_windows.ps1 -WorkspaceDir E:\GPTProject2\vLLM
powershell -ExecutionPolicy Bypass -File .\scripts\healthcheck_openai.ps1 -WorkspaceDir E:\GPTProject2\vLLM
powershell -ExecutionPolicy Bypass -File .\scripts\ask_local_openai.ps1 -WorkspaceDir E:\GPTProject2\vLLM -Message "只回答：llama.cpp 本机调用成功。" -MaxTokens 16 -Timeout 120
```

Current default profile details:
- profile: `qwen3_8b_gguf_q4_k_m`
- model alias: `Qwen3-8B-GGUF-q4_k_m-local`
- local port: `8001`

Remote ask from another Windows machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ask_remote_openai.ps1 -WorkspaceDir E:\GPTProject2\vLLM -HostName YOUR_LAPTOP_IP -Port 8001 -ApiKey change-this-before-lan-use -Model Qwen3-8B-GGUF-q4_k_m-local -Message "只回答：远程 llama.cpp 调用成功。"
```

If you want LAN access on `8001`, you only need a firewall rule. Native `llama.cpp` does not need the old WSL `portproxy`.

## Notes
- Stop the old `vLLM` service before starting `llama.cpp` on the same port.
- The current profile uses the first GGUF shard as `MODEL_ENTRY_FILE`; `llama.cpp` resolves the rest of the split files by filename convention.
