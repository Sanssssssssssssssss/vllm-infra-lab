# ARCHITECTURE

## 文档状态
状态：Draft v0.1

## 架构目标
构建一个本地运行的大模型推理服务，使远端 agent 可以通过统一 API 调用本机模型，并为后续量化、LoRA、tracing 等实验保留扩展空间。

## 当前架构原则
- 先跑通单模型推理服务，再做性能与训练相关扩展
- 优先采用标准化 API，降低 agent 接入成本
- 配置外置化，避免硬编码
- 将“服务运行”与“实验扩展”边界分离

## 初版技术方向
- 运行时：`vLLM`
- 服务接口：OpenAI 兼容 HTTP API
- 模型：`Qwen3-8B`
- 宿主路径：`Windows 11 + WSL2 Ubuntu`
- 配置：环境变量文件
- 管理脚本：PowerShell / Python / WSL Bash
- 验证方式：健康检查 + 样例推理请求 + 基线记录

## 建议模块
### 1. runtime
负责模型服务启动、停止、重启、运行参数管理。

当前落地：
- `scripts/bootstrap_vllm_wsl.ps1`
- `scripts/start_vllm_wsl.ps1`

### 2. config
负责保存主机地址、端口、模型路径、运行参数等配置。

当前落地：
- `config/runtime.env`
- `config/runtime.env.example`

### 3. api-validation
负责健康检查、样例请求、响应校验。

当前落地：
- `scripts/healthcheck_vllm.ps1`
- `scripts/test_openai_api.py`

### 4. benchmark
负责记录基础性能指标，为后续优化做对比。

### 5. docs
负责部署说明、调用说明、故障排查说明。

当前落地：
- `docs/V1_DEPLOYMENT.md`

## 部署拓扑
- 宿主机：本地笔记本，运行模型服务
- 调用端：另一台电脑上的 agent 系统
- 通信方式：局域网内 HTTP 调用

## 关键数据对象
### ModelProfile
- model_name
- model_path
- quantization_mode
- max_model_len
- dtype

### RuntimeConfig
- host
- port
- gpu_memory_utilization
- tensor_parallel_size
- max_num_seqs
- api_key_enabled

### BenchmarkRecord
- test_name
- timestamp
- prompt_size
- output_size
- first_token_latency
- total_latency
- tokens_per_second
- notes

## 接口原则
- 优先保持与常见 agent 调用方式兼容
- 输入输出格式尽量稳定，减少后续频繁改动
- 明确区分配置错误、环境错误、运行错误

首版接口范围：
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## 当前待确认项
- 首版 CPU offload 参数是否需要进一步调小或调大
- 首版是否需要额外加反向代理或只保留 API key
- AWQ 切入点与基线指标门槛
## 2026-03-15 Architecture Update
- Shared API settings now live in `config/runtime.env`
- Backend- and model-specific settings now live in `config/profiles/*.env`
- The workspace currently maintains both a `vllm` profile and a `llamacpp` profile
- Windows-native fast experiments should use the `llama.cpp` profile instead of rewriting the vLLM baseline
