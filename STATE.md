# STATE

## 当前状态
项目已完成初始化文档与首版部署骨架搭建。

当前阶段：
- 已明确项目主目标是本地大模型部署与远程 API 调用
- 已确定首个目标模型为 `Qwen3-8B`
- 已将量化、LoRA、蒸馏、tracing 归入后续演进方向
- 已确认首版运行路径为 `Windows 11 + WSL2 Ubuntu + vLLM`
- 已生成配置模板、部署说明、环境检查脚本、启动脚本、验证脚本
- 已完成 WSL Ubuntu 安装、vLLM 依赖安装、本地模型下载和本机服务启动
- 已验证本机 `localhost` 上的 OpenAI 风格 API 调用闭环
- 已建立 `log.md`，用于记录 Linux / WSL 下已验证成功的 CLI
- 已将默认操作逻辑切换为 WSL-first
- 已确认共享 GPU 内存不应作为当前路径的性能优化目标
- 已为服务配置固定模型别名 `Qwen3-8B-local`
- 当前尚未打通局域网访问，因为当前会话没有管理员权限配置 Windows portproxy 和防火墙规则

## 当前项目目标
完成首个可运行版本，使另一台电脑上的 agent 可以调用本机模型完成推理。

## 下一步
1. 用管理员 PowerShell 添加 Windows portproxy
2. 用管理员 PowerShell 添加防火墙入站规则
3. 从另一台电脑验证局域网调用
4. 建立首版性能基线并收敛参数

## 风险
- 当前硬件下 `Qwen3-8B` 的显存/内存适配空间有限
- CPU offload 虽能提高首版跑通概率，但可能显著影响吞吐和延迟
- 若进一步增加 offload，容量可能更宽松，但性能通常会更差
- 若过早进入量化/微调，会分散首版闭环交付
- Windows 防火墙和局域网环境可能影响跨机器访问
- 当前保守配置下生成速度很慢，本机实测约 0.6-0.7 tokens/s

## 不确定点
- `Qwen3-8B` 在当前机器上的最终可用参数边界
- 是否只靠 API key 即可满足首版安全边界
- AWQ 是否会成为首个必须推进的优化步骤
- 是否要把 `enable_thinking=False` 作为 agent 场景的默认请求参数

## 接班说明
- 每轮开始先读取本文件以及其余项目记忆文件重建上下文
- 若需求与聊天内容冲突，以最新确认的项目文档为准，并显式指出
- 新的架构或技术取舍必须更新 `DECISIONS.md`
- 新的任务推进必须同步更新 `TASKS.md` 与 `STATE.md`
- 新的成功 Linux CLI 应同步追加到 `log.md`

## 2026-03-15 更新
- 已确认另一台电脑的局域网调用验证成功；此前“局域网尚未打通”的状态已过期
- 当前主要瓶颈已从“能否访问”转为“单请求吞吐过低”
- 已完成一个最小原生 Windows 可行性实验：在 Windows 侧无法为 `vllm==0.17.1` 获取官方 `win_amd64` wheel，因此当前无法做等价的原生 Windows vLLM 性能对比
- 当前性能优化优先级应转向量化路线（如 AWQ）和运行参数收敛，而不是继续增加 offload 或追求吃满共享 GPU 内存
## 2026-03-15 Profile Update
- Active profile support has been added through `config/profiles/*.env`
- The current default profile is `qwen25_7b_instruct_gguf_q4_k_m`
- Official `llama.cpp` Windows CUDA binaries are ready under `E:\GPTProject2\tools\llama.cpp\b8352\win-cuda-12.4-x64`
- The first GGUF shard has been downloaded successfully; the second shard should now be continued via the Hugging Face CLI cache-copy path
- `Qwen3-8B-GGUF q4_k_m` has now been downloaded and verified locally with `llama.cpp`
- Current local native endpoint: `http://127.0.0.1:8001`
- Current validated model alias: `Qwen3-8B-GGUF-q4_k_m-local`
- Current observed short-answer generation speed is about `44 tokens/s`, which is far above the previous WSL + vLLM baseline

## 2026-05-05 vLLM Optimization Update
- Reconfirmed WSL GPU availability and `vllm==0.17.1`.
- Started a new WSL + vLLM optimized route using the already downloaded `Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`.
- Validated local OpenAI-compatible chat on `http://127.0.0.1:8000` with model alias `Qwen3-8B-GGUF-vLLM-local`.
- Enabled the first optimization set: FlashAttention auto-selection, automatic prefix caching, chunked prefill, async scheduling, `max-num-seqs=8`, `max-num-batched-tokens=4096`, and CUDA graph capture.
- Current working `gpu_memory_utilization` is `0.80`; `0.86` failed at startup because WSL free VRAM was slightly below the requested allocation.
- Startup logs reported about `4.82 GiB` model memory, about `1.13 GiB` available KV cache memory, `8,224` GPU KV cache tokens, and about `4.02x` maximum concurrency at `2048` tokens per request.
- A 4-request local concurrency smoke test completed successfully; this is not yet a formal benchmark.

## 2026-05-05 Benchmarking Update
- Added `scripts/bench_openai_async.py` as the standard OpenAI-compatible streaming benchmark runner.
- Added `docs/BENCHMARKING.md` and `docs/EXPERIMENTS.md` as the required benchmark and experiment loop contract.
- Future tuning rounds must record benchmark CSV/JSONL artifacts and push each completed round to the GitHub repository before continuing.
- Ran the first required `vllm + gguf-q4_k_m` matrix with `short_chat`, `long_prefill`, `long_decode`, and `shared_prefix` at concurrency `1,2,4,8`.
- Results are stored in `reports/benchmarks/2026-05-05-vllm-gguf-matrix.csv` and `.jsonl`; all 16 matrix rows completed with `error_count=0`.
