# TASKS

## 待办 Backlog
- [ ] 评估 `Qwen3-8B` 在当前硬件上的基础可行性与参数边界
- [ ] 建立基础 benchmark 记录格式
- [ ] 规划 AWQ / LoRA / tracing 后续入口
- [ ] 实测并收敛 `VLLM_CPU_OFFLOAD_GB`、`VLLM_MAX_NUM_SEQS`、`VLLM_MAX_MODEL_LEN`
- [ ] 记录首版启动日志与首个成功推理样例
- [ ] 用管理员 PowerShell 配置 Windows portproxy 与防火墙规则
- [ ] 验证另一台电脑的真实局域网调用链路
- [ ] 评估关闭 thinking 模式是否应作为 agent 默认配置
- [x] 为 vLLM GGUF 优化路线建立 benchmark 表格与请求压测脚本
- [ ] 系统比较 prefill / KV cache / continuous batching / automatic prefix caching 参数矩阵
- [ ] 运行完整 Stage 2 显存/KV cache 288-start sweep

## 进行中 In Progress
- [ ] 收敛首版局域网暴露方案（WSL portproxy + firewall）

## 已完成 Done
- [x] 建立项目长期记忆机制约束
- [x] 产出首版项目目标、需求、架构、决策文档
- [x] 完成项目初始化与首版执行计划
- [x] 确认宿主环境为 Windows 11 + NVIDIA RTX 4070 Laptop + 32 GB RAM
- [x] 确认首版部署路径为 `WSL2 Ubuntu + vLLM`
- [x] 确认首版 API 目标为最小 OpenAI 兼容子集
- [x] 落地配置模板、部署文档、环境检查脚本、启动脚本、验证脚本
- [x] 安装 Ubuntu for WSL2 并确认 WSL 内 GPU 可用
- [x] 在 WSL 中安装 vLLM 0.17.1 与依赖
- [x] 将 `Qwen3-8B` 下载到本地目录
- [x] 跑通首个模型服务启动
- [x] 完成本机 `GET /health`、`GET /v1/models` 验证
- [x] 完成本机 OpenAI 风格聊天请求验证
- [x] 识别并绕过 `hf_xet` 下载卡住问题
- [x] 识别并绕过 `torch.compile`/`torch._dynamo` 与 CPU offload 的兼容问题
- [x] 建立 Linux CLI 成功路径日志 `log.md`
- [x] 配置 Windows portproxy 与防火墙规则并完成另一台电脑的局域网调用验证
- [x] 用 vLLM 直接加载本地 `Qwen3-8B-GGUF Q4_K_M` 并完成本机聊天验证
- [x] 为 vLLM GGUF 优化路线落地可复用启动与提问脚本

## 阻塞 Blocked
- [ ] 当前会话没有 Windows 管理员权限，无法直接配置 portproxy 和防火墙规则

## 建议优先级
1. 先确认运行环境与首版接口目标
2. 跑通本地服务
3. 用管理员权限打通局域网访问
4. 建立基线后再进入性能优化与实验扩展

## 2026-03-15 新增待办
- [ ] 对 `Qwen3-8B` 的量化路线做首轮评估，优先比较 AWQ 是否能显著降低当前 0.6-0.7 tokens/s 的延迟问题
- [ ] 记录“当前 WSL + fp16 + CPU offload”基线配置与吞吐，作为后续量化实验对照
- [ ] 如需评估原生 Windows 低延迟路线，单独立项比较 `Ollama / llama.cpp / 其他原生栈`，不与当前 vLLM 首版混为一条线
## 2026-03-15 llama.cpp Track
- [x] Add profile-driven backend switching scaffolding
- [x] Download and extract the official `llama.cpp` Windows CUDA binaries
- [x] Download `Qwen3-8B-GGUF q4_k_m`
- [x] Start `llama.cpp` and validate the local OpenAI-compatible API for the active profile
- [ ] Add a firewall rule for native Windows `llama.cpp` on port `8001` if LAN access is needed

## 2026-05-05 vLLM Optimization Track
- [x] Reconfirm WSL vLLM runtime and GPU visibility
- [x] Start `Qwen3-8B-GGUF Q4_K_M` through vLLM on port `8000`
- [x] Validate OpenAI-compatible chat against `Qwen3-8B-GGUF-vLLM-local`
- [x] Capture initial startup facts for prefix caching, chunked prefill, KV cache capacity, continuous batching, and CUDA graph
- [x] Add a repeatable concurrency benchmark harness
- [x] Run the first required workload/concurrency benchmark matrix and persist CSV/JSONL artifacts
- [x] Add startup-level vLLM memory/KV cache profiler
- [x] Run Stage 2 memory profiling pilot and persist CSV/JSONL artifacts
- [ ] Compare warm-prefix and cold-prefix request behavior
