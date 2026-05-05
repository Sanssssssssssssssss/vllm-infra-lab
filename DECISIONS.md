# DECISIONS

## ADR-001 使用项目文档作为长期记忆
状态：Accepted

决策：
使用 `PROJECT_BRIEF.md`、`REQUIREMENTS.md`、`ARCHITECTURE.md`、`DECISIONS.md`、`TASKS.md`、`STATE.md` 作为长期项目记忆。

原因：
- 聊天上下文不可靠，无法作为长期唯一记忆
- 需要在多轮会话中稳定恢复项目状态

## ADR-002 首阶段聚焦单模型可运行闭环
状态：Accepted

决策：
首阶段先完成单模型本地部署、API 暴露、远程调用验证，不同时并行推进量化、微调、蒸馏。

原因：
- 当前最重要的是建立可运行基线
- 在缺少稳定基线前做高级优化会增加排查复杂度

## ADR-003 首个目标模型为 Qwen3-8B
状态：Accepted

决策：
首个验证模型固定为 `Qwen3-8B`。

原因：
- 用户已明确当前想先尝试该模型
- 便于围绕单一目标建立部署、性能和兼容性基线

## ADR-004 首选 vLLM 方向
状态：Accepted

决策：
项目首版采用 `vLLM` 作为推理服务方案。

原因：
- 用户明确希望本目录用于本地大模型部署和 API 实现
- `vLLM` 方向契合高性能推理探索目标

## ADR-005 首版默认局域网访问
状态：Accepted

决策：
首版默认服务面向局域网内另一台电脑访问，不以公网暴露为目标。

原因：
- 与当前使用场景一致
- 能降低安全和运维复杂度

## ADR-006 首版运行环境采用 Windows 11 + WSL2 Ubuntu
状态：Accepted

决策：
首版运行环境采用 `Windows 11` 作为宿主系统，使用 `WSL2 Ubuntu` 运行 `vLLM` 服务，而不是 Windows 原生直接运行。

原因：
- 当前机器已具备 Windows、NVIDIA 驱动和 WSL 能力
- `vLLM` 官方预编译 CUDA wheel 面向 Linux x86_64
- 该路径在当前机器上改造成本最低

## ADR-007 首版 API 采用最小 OpenAI 兼容子集
状态：Accepted

决策：
首版 API 以最小必要 OpenAI 兼容接口为目标，优先覆盖：
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

原因：
- 能满足另一台电脑上的 agent 集成需要
- 可以避免首阶段扩展过多接口

## ADR-008 首版允许 CPU Offload 作为可运行兜底
状态：Accepted

决策：
在当前 8 GB 显存约束下，首版允许使用 `cpu-offload-gb` 作为 `Qwen3-8B` 的可运行兜底方案。

原因：
- 保持首个目标模型不变
- 优先建立可运行基线，再推进 AWQ 等性能导向方案

## ADR-009 首版模型先下载到本地目录
状态：Accepted

决策：
首版先将 `Qwen3-8B` 下载到本地目录 `/mnt/e/GPTProject2/models/Qwen3-8B`，服务从本地路径加载，而不是每次启动时通过远程仓库拉取。

原因：
- 远程下载和服务启动解耦后更容易排障
- 可以避免 WSL 下 Xet 下载链路带来的不确定性

## ADR-010 首版启用 enforce-eager
状态：Accepted

决策：
首版运行参数启用 `--enforce-eager`。

原因：
- 当前 `CPU offload + torch.compile` 组合在本机环境下触发了 `torch._dynamo` 跟踪失败
- eager 模式牺牲部分性能，但能优先保证服务稳定启动

## ADR-011 运行工作流以 WSL 终端为主
状态：Accepted

决策：
首版运行工作流默认以 WSL Ubuntu 终端为主入口，模型、环境、启动、验证等日常命令优先使用 `.sh` 脚本；只有必须依赖 Windows 管理员权限的网络暴露步骤，才从 WSL 间接拉起 Windows 提权窗口。

原因：
- 更符合用户偏好的工作方式
- 与 vLLM 的 Linux 运行路径一致
- 能减少在 PowerShell 与 WSL 之间频繁切换

## ADR-012 不以“吃满共享 GPU 内存”作为性能目标
状态：Accepted

决策：
当前 `Windows + WSL2 + CUDA + vLLM` 路径下，不以“吃满 Windows 任务管理器里的共享 GPU 内存”作为优化目标。性能优化优先级应放在：
- 降低 CPU offload 依赖
- 尝试 AWQ 等量化
- 收敛可提高 tokens/s 的运行参数

原因：
- `cpu_offload_gb` 本质上使用的是 CPU / 系统内存，vLLM 官方说明这是用 CPU memory 扩展容量，而不是等价 GPU memory
- NVIDIA 的 CUDA on WSL 文档说明，WSL 不支持完整 Unified Memory，且 pinned system memory 可用性受限
- 因此，共享内存更适合作为“能跑起来”的容量兜底，而不是性能优化目标

## ADR-013 当前不推进原生 Windows vLLM 作为首版优化方向
状态：Accepted

决策：
当前首版不把“脱离 WSL、直接在 Windows 原生运行 vLLM”作为性能优化主线。若后续要评估原生 Windows 推理，应视为新的服务栈实验，而不是当前 vLLM 路线的就地优化。

原因：
- vLLM 官方安装与发布路径仍以 Linux x86_64 CUDA 为主，当前仓库已确认的运行方案也是 `Windows 11 + WSL2 Ubuntu + vLLM`
- 本机在 Windows 侧执行 `pip download --only-binary=:all: --platform win_amd64 ... vllm==0.17.1` 实测无法获取官方 wheel
- 因此，当前无法做“同一套 vLLM 服务在原生 Windows 下是否更快”的等价对比实验
- 若要追求原生 Windows 下更低延迟，更可行的路线通常是改用其他推理栈并配合量化，这会超出当前首版架构范围
## ADR-014 Use Profile-Driven Backend Switching
状态：Accepted

决策：
Use `config/profiles/*.env` for model/backend profiles and `config/runtime.env` with `ACTIVE_PROFILE` as the single switch.

原因：
- model switching will be a normal workflow
- the workspace must keep both the original vLLM route and newer Windows-friendly routes
- profile files are easier to audit than hand-editing scattered launch parameters

## ADR-015 Current Fast Path Uses llama.cpp CUDA + Qwen2.5-7B-Instruct-GGUF q4_k_m
状态：Accepted

决策：
For the current speed-oriented experiment, adopt `Windows native + llama.cpp CUDA + Qwen/Qwen2.5-7B-Instruct-GGUF + q4_k_m`, while keeping the `qwen3_8b_vllm` entry available.

原因：
- the user explicitly asked for a Windows-friendly llama.cpp CUDA route
- GGUF quantization is a better fit for the current 8 GB laptop GPU
- this route is more aligned with single-request latency optimization than the current fp16 + offload vLLM baseline

## 2026-03-15 Update
- The active fast-path has now been moved from `Qwen2.5-7B-Instruct-GGUF` to `Qwen3-8B-GGUF q4_k_m`
- The Windows-native `llama.cpp` route is verified locally on port `8001`

## ADR-016 Optimized vLLM Experiments Use Local Qwen3 GGUF First
状态：Accepted

决策：
For the next vLLM-focused optimization track, use the already downloaded `Qwen3-8B-GGUF Q4_K_M` file with vLLM's GGUF loader before returning to fp16 safetensors + CPU offload.

原因：
- The previous fp16 safetensors baseline required CPU offload and `--enforce-eager`, which made it a poor starting point for studying prefill scheduling, KV cache behavior, CUDA graph, and continuous batching.
- The local GGUF model fits the 8 GB laptop GPU much better and lets vLLM run without CPU weight offload.
- This route keeps the work inside `WSL2 + vLLM`, matching the user's AI infra learning goal, while still using a model artifact already present on disk.

## ADR-017 Benchmark Artifacts Are Required Before Further Tuning
状态：Accepted

决策：
All future inference tuning should use the standard OpenAI-compatible benchmark runner and persist CSV/JSONL artifacts under `reports/benchmarks/`. Each completed round must be pushed to the GitHub repository before moving to the next tuning step.

原因：
- Smoke tests and manual log watching are insufficient for comparing prefill, KV cache, continuous batching, and prefix caching changes.
- TTFT, ITL, aggregate TPS, E2E latency, GPU memory, and error counts need stable schemas so results can be compared across runs.
- Keeping benchmark rules in the repository makes the learning trail auditable and reproducible.
