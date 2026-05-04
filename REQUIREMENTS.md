# REQUIREMENTS

## 文档状态
状态：Draft v0.1

## 需求来源
- 用户希望在本地笔记本部署大模型，并通过 API 供另一台电脑上的 agent 调用
- 当前目标模型：`Qwen3-8B`
- 中长期希望探索 AWQ、蒸馏、LoRA、tracing，并尽量压榨 GPU 性能

## 缺失信息
- 性能验收阈值未最终确认
- 首版成功后切入 AWQ 的时点与评估标准未最终确认

## 当前采用的假设
- 首版以局域网内访问为主
- 首版采用 OpenAI 兼容风格 HTTP API
- 首版先保证可运行与可复现，再进入量化和微调实验
- 首版默认只支持一个主模型配置
- 首版运行路径为 `Windows 11 + WSL2 Ubuntu + vLLM`

## 功能需求
### FR-001 环境初始化
系统必须能够在目标宿主机上完成依赖安装、模型准备和运行配置初始化。

验收标准：
- 有明确的环境准备步骤
- 有可执行的启动前检查项
- 新机器按文档可完成初始化

### FR-002 模型加载
系统必须能够加载 `Qwen3-8B` 并提供推理能力。

验收标准：
- 模型可成功启动
- 模型可处理至少一个文本生成请求
- 启动失败时可定位主要原因

### FR-003 推理 API
系统必须提供可从另一台电脑访问的推理 API。

验收标准：
- 可通过 HTTP 发起请求并收到模型响应
- API 输入输出格式被文档化
- 至少有一个可复现的调用示例
- 首版至少支持 `GET /health`、`GET /v1/models`、`POST /v1/chat/completions`

### FR-004 远程访问
系统必须支持另一台电脑在预期网络范围内访问本地模型服务。

验收标准：
- 通过局域网 IP 或约定地址成功访问
- 网络绑定地址、端口、访问限制清晰可配置
- 有基本安全边界说明
- 若默认防火墙阻断，有明确放行步骤

### FR-005 服务管理
系统必须支持启动、停止、重启和健康检查。

验收标准：
- 提供统一启动方式
- 可检测服务是否在线
- 常见故障有基础排查方法

### FR-006 配置管理
系统必须把关键运行参数配置化，避免硬编码。

验收标准：
- 模型路径、端口、主机地址、并发相关参数可配置
- 配置修改后无需改动业务代码
- 配置项有说明
- 至少支持模型名、API key、CPU offload、GPU memory utilization、max model len 的配置

### FR-007 基线验证
系统必须支持基础验证与记录，为后续优化提供基线。

验收标准：
- 至少记录一次可复现的启动与推理验证结果
- 至少定义一组基础性能观察指标
- 可区分“功能跑通”和“性能优化”两个阶段

### FR-008 可扩展实验入口
系统必须为后续 AWQ、LoRA、蒸馏、tracing 留出清晰扩展点。

验收标准：
- 文档中明确哪些模块未来可扩展
- 当前实现不会阻断后续量化/微调实验

## 非功能需求
### NFR-001 简单性
优先简单、稳健、可维护的方案，不引入无必要的复杂组件。

### NFR-002 可复现性
环境准备、启动步骤、验证方式必须可重复执行。

### NFR-003 资源适配
方案必须考虑 8 GB 专享显存和 32 GB 系统内存的现实限制。

### NFR-004 可观测性
至少具备基础日志、错误输出和健康状态检查能力。

### NFR-005 安全边界
首版至少要限制在预期网络范围内使用，避免无控制暴露。

### NFR-006 可演进性
架构应支持后续加入量化、微调、追踪与性能对比实验。

## 业务规则
- BR-001：未经确认，不新增与当前目标无关的功能
- BR-002：所有实现必须映射到本文件中的需求
- BR-003：首版优先建立“单模型 + 单服务 + 可远程调用”的闭环
- BR-004：优化工作必须基于可运行基线推进
- BR-005：后续架构或技术选型变更必须记录到 `DECISIONS.md`

## 当前不纳入验收范围
- 公网部署
- 多租户鉴权
- 多模型调度
- 完整训练平台
- 自动化实验编排系统
## 2026-03-15 Requirement Addendum
### FR-009 Model Profile Management
The system must support switching models and inference backends through externalized profile files instead of manual script edits.

Acceptance:
- At least one `vllm` profile and one `llamacpp` profile are available
- The active profile can be switched from a single command
- Model path, backend type, download parameters, and launch parameters are all profile-driven
