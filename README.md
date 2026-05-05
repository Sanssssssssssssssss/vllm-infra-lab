# Local LLM Deployment Workspace

This repository is the deployment workspace for local LLM inference backends and model profiles.

Current focus:
- Keep the original `Qwen3-8B + vLLM + WSL2` baseline available
- Add an optimized `Qwen3-8B GGUF + vLLM + WSL2` route for infra experiments
- Add a Windows-friendly `llama.cpp + CUDA + GGUF` route for faster single-request experimentation
- Expose an OpenAI-compatible API over LAN for another computer's agent

Profile-driven routes:
- `qwen3_8b_vllm`: Windows 11 host + WSL2 Ubuntu + `vLLM` + `Qwen/Qwen3-8B`
- `qwen3_8b_gguf_vllm_optimized`: WSL2 Ubuntu + `vLLM` + local `Qwen3-8B-GGUF Q4_K_M` with prefix caching/chunked prefill/continuous batching knobs
- `qwen25_7b_instruct_gguf_q4_k_m`: Windows 11 native + `llama.cpp` CUDA + `Qwen/Qwen2.5-7B-Instruct-GGUF`

Start here:
- Read [docs/LLAMACPP_WINDOWS.md](E:\GPTProject2\vLLM\docs\LLAMACPP_WINDOWS.md) for the Windows-native path
- Read [docs/V1_DEPLOYMENT.md](E:\GPTProject2\vLLM\docs\V1_DEPLOYMENT.md) for the original WSL + vLLM path
- Read [docs/VLLM_OPTIMIZATION.md](E:\GPTProject2\vLLM\docs\VLLM_OPTIMIZATION.md) for the optimized WSL + vLLM GGUF path
- Read [docs/BENCHMARKING.md](E:\GPTProject2\vLLM\docs\BENCHMARKING.md) for the benchmark runner and required metrics
- Read [docs/EXPERIMENTS.md](E:\GPTProject2\vLLM\docs\EXPERIMENTS.md) for the experiment loop and tuning matrix
- Read [reports/2026-05-05-vllm-gguf-optimization.md](E:\GPTProject2\vLLM\reports\2026-05-05-vllm-gguf-optimization.md) for the first vLLM GGUF optimization report
- List profiles with [scripts/list_profiles.ps1](E:\GPTProject2\vLLM\scripts\list_profiles.ps1)
- Switch the active profile with [scripts/set_active_profile.ps1](E:\GPTProject2\vLLM\scripts\set_active_profile.ps1)
- Start the optimized vLLM GGUF route with [scripts/start_vllm_gguf_optimized_wsl.ps1](E:\GPTProject2\vLLM\scripts\start_vllm_gguf_optimized_wsl.ps1)
- Ask the optimized vLLM GGUF route with [scripts/ask_vllm_gguf_optimized_wsl.sh](E:\GPTProject2\vLLM\scripts\ask_vllm_gguf_optimized_wsl.sh)
- Run OpenAI-compatible async benchmarks with [scripts/bench_openai_async.py](E:\GPTProject2\vLLM\scripts\bench_openai_async.py)
- Bootstrap `llama.cpp` with [scripts/bootstrap_llamacpp_windows.ps1](E:\GPTProject2\vLLM\scripts\bootstrap_llamacpp_windows.ps1)
- Download the active GGUF model with [scripts/download_active_model_windows.ps1](E:\GPTProject2\vLLM\scripts\download_active_model_windows.ps1)
- Start `llama.cpp` with [scripts/start_llamacpp_windows.ps1](E:\GPTProject2\vLLM\scripts\start_llamacpp_windows.ps1)
- Ask the active local backend with [scripts/ask_local_openai.ps1](E:\GPTProject2\vLLM\scripts\ask_local_openai.ps1)
