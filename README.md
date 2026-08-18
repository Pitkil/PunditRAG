# PunditRAG

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

面向中文技术资料的可追溯 RAG 知识库系统。PunditRAG 将文档解析、结构化切分、混合检索、HyDE、RRF、重排、引用生成和可复现评测串成一条完整链路，并提供可直接使用的知识库工作台与 REST API。

> 当前重点不是“让模型尽量回答”，而是让回答能够被检索、引用和评测：没有足够证据时拒绝作答，型号、编号、数值和日期必须来自引用来源。

## 目录

- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [对话处理过程](#对话处理过程)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用方式](#使用方式)
- [API 概览](#api-概览)
- [Prompt 设计](#prompt-设计)
- [评测结果](#评测结果)
- [测试](#测试)
- [安全与可靠性](#安全与可靠性)
- [项目结构](#项目结构)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 核心能力

### 文档导入

- 使用 MinerU API 将 PDF 转换为 Markdown。
- 支持直接导入 `.md`，并可将 `.txt`、`.docx`、`.pptx`、`.xlsx`、`.csv`、`.html`、`.htm`、`.json` 转换为 Markdown。
- 按 Markdown 标题、段落、表格和表格行进行结构化切分。
- 默认以近似 token 计数控制 `500` token 切片和 `80` token 重叠。
- 对密集的“字段 + 参数”技术指标按行分组，并在每个子切片中保留章节标题。
- Embedding 文本包含文档名、章节名和主题信息，降低脱离上下文的误召回。
- 切片向量写入 Milvus，知识库、文档与会话元数据写入 MongoDB，Markdown 图片处理路径中的图片写入 MinIO；导入中间文件保存在 `temp-files/`。

### 检索与回答

- BGE-M3 Dense + Sparse 混合召回。
- 原始问题检索与 HyDE 假设文档检索并行执行。
- 使用 RRF 融合多路结果，再通过 BGE Reranker 精排。
- 主题匹配作为召回扩展，不作为可能漏召回的硬过滤条件。
- 高于重排阈值的候选进入常规回答；全部低于阈值但确有召回时，保留少量低置信候选交给回答模型核验，而不是在重排阶段提前拒答。
- 只有本地与可选联网检索都没有返回候选时，才直接给出无资料拒答。
- 查询范围显式区分全部知识库、指定知识库和指定文档；文档范围会生成精确的 `document_id` 过滤条件。
- 总结和深度讲解可进入全文综合：短文档直接把完整上下文交给模型，长文档使用 Map/Reduce 分批处理。
- “继续”“更详细一点”“展开”等追问可从上一轮引用来源继承文档范围。
- 跨网页与本地资料去重时优先保留本地原文；回答使用连续的 `[n]` 引用，API 只返回实际引用的证据。
- 联网补充默认关闭；显式开启后，联网失败只降级联网分支，不影响本地知识库检索。

### 工作台与运维

- 知识库、文档和会话管理。
- 非流式响应与 SSE 流式响应。
- MongoDB、Milvus、MinIO 真实依赖健康检查。
- 文档删除时同步清理向量、对象存储、本地文件和元数据。
- 自建集及 RGB、CRUD-RAG、MTRAG 评测适配器。

## 系统架构

### 文档导入架构

```mermaid
flowchart LR
    U[浏览器 / API 客户端] -->|上传文件| I1[上传、校验与任务创建]
    I1 -->|创建文档与任务记录| M[(MongoDB)]
    I1 -->|PDF| IP[PDF / MinerU 转换]
    I1 -->|Office / 文本| IC[通用文件转换]
    I1 -->|Markdown| I3[图片解析与地址替换]
    IP -->|MinerU Markdown| I4[结构化切分]
    IC -->|转换后的 Markdown| I3
    I3 <-->|图片与摘要| L[OpenAI-compatible VL Model]
    I3 -->|上传解析图片| O[(MinIO)]
    I3 -->|规范化 Markdown| I4
    I4 -->|带标题和元数据的切片| I5[文档主题识别]
    I5 <-->|主题识别 Prompt / 名称| LLM[OpenAI-compatible LLM]
    I5 -->|主题向量| V[(Milvus)]
    I5 -->|附带主题的切片| I6[BGE-M3 Dense / Sparse Embedding]
    I6 -->|切片向量与元数据| V
```

### 查询与回答架构

```mermaid
flowchart TB
    U[浏览器 / API 客户端] -->|POST /query| QA[范围校验与运行状态]
    QA -->|校验知识库 / 文档并维护会话| M[(MongoDB)]
    QA -->|初始化 QueryGraphState| Q1[历史读取、问题重写与查询计划]
    M -->|最近对话历史| Q1
    Q1 <-->|问题改写 Prompt / 独立问题| L[OpenAI-compatible LLM]
    Q1 -->|普通问答| Q2[原问题混合检索]
    Q1 -->|普通问答| Q3[HyDE 混合检索]
    Q1 -->|enable_web_search=true| Q4[Web Search]
    Q1 -->|总结或深度讲解| QS[全文综合]
    Q2 <-->|Dense / Sparse 查询与候选| V[(Milvus)]
    Q3 <-->|生成 HyDE 文本| L
    Q3 <-->|Dense / Sparse 查询与候选| V
    Q4 <-->|搜索请求与网页结果| W[Web Search MCP]
    Q2 -->|原问题排序列表| Q5[RRF 融合]
    Q3 -->|HyDE 排序列表| Q5
    Q5 -->|本地融合候选| Q6[BGE Reranker 节点]
    Q4 -->|联网候选| Q6
    Q6 <-->|问题-证据对与相关性分数| R[BGE Reranker]
    Q6 -->|分级后的 Top-K 证据| Q7[证据约束回答与结果过滤]
    QS <-->|读取摘要范围| V
    QS <-->|短文档完整上下文或长文档 Map / Reduce| L
    QS -->|摘要结果| Q7
    Q7 <-->|回答 Prompt 与带引用答案| L
    Q7 -->|保存回答、来源与图片| M
    Q7 -->|完整结果或 SSE 事件| QA
    QA -->|JSON / SSE| U
```

### 导入链路箭头说明

| 箭头 | 传递内容 | 作用 |
|---|---|---|
| 客户端 → 上传、校验与任务创建 | 文件、`kb_id` | 校验知识库、扩展名、文件名、数量和大小，创建独立导入任务。 |
| 上传、校验与任务创建 → MongoDB | 知识库、文档和任务元数据 | 记录 `document_id`、文件名、状态、错误与切片数量，供工作台查询。 |
| 上传、校验与任务创建 → PDF / MinerU 转换 | `.pdf` 文件路径 | 申请 MinerU 上传地址、轮询解析任务、下载并解压结果，得到 Markdown。 |
| 上传、校验与任务创建 → Office / 文本转换 | `.txt`、`.docx`、`.pptx`、`.xlsx`、`.csv`、`.html`、`.htm`、`.json` 文件路径 | 在本地抽取正文、表格等内容并生成统一 Markdown。 |
| 上传、校验与任务创建 → 图片解析 | 原生 `.md` 文件路径 | Markdown 无需内容转换，直接检查其同目录图片资源。 |
| PDF / MinerU 转换 → 结构化切分 | MinerU 返回的 Markdown | 当前 PDF 路径直接进入切分，不经过 Markdown 图片摘要与 MinIO 上传节点。 |
| Office / 文本转换 → 图片解析 | 转换后的 Markdown | 复用 Markdown 图片处理节点；没有图片时直接透传正文。 |
| 图片解析 ↔ VL Model | 图片、相邻正文 ↔ 单行客观摘要 | 为图片生成可检索的替代文本；无法识别时写入固定失败提示，不虚构内容。 |
| 图片解析 → MinIO | 从 Markdown 发现的本地图片 | 上传到私有 Bucket，并用受控资源地址替换本地路径。 |
| 图片解析 → 结构化切分 | 图片地址已规范化的 Markdown | 保证后续切片携带可访问的图片上下文。 |
| 结构化切分 → 文档主题识别 | 标题、段落、表格及切片元数据 | 按标题和近似 token 预算切分，密集参数表按行分组并保留标题。 |
| 文档主题识别 → LLM | 文件标题和前若干切片 | 请求模型识别文档的规范主题名称。 |
| LLM → 文档主题识别 | 单行主题名称 | 将主题写入每个切片；无法识别时回退到文件标题。 |
| 文档主题识别 → Milvus | 主题名称的 Dense/Sparse 向量 | 建立资料主题索引，用于查询阶段的召回扩展。 |
| 文档主题识别 → BGE-M3 Embedding | 带文档名、章节名和主题的切片 | 为切片补足上下文后批量生成向量。 |
| BGE-M3 Embedding → Milvus | 切片正文、元数据、Dense/Sparse 向量 | 写入知识库切片集合，作为本地混合检索的数据源。 |

### 查询链路箭头说明

| 箭头 | 传递内容 | 作用 |
|---|---|---|
| 客户端 → 请求校验与运行状态 | 问题、`session_id`、`scope_mode`、`kb_ids`、`document_ids`、流式与联网开关 | 创建本轮 `run_id`，把“全部知识库 / 指定知识库 / 指定文档”解析成明确范围，并选择同步或后台执行。 |
| 请求校验与运行状态 → MongoDB | 会话 ID、首轮问题、知识库 ID、文档 ID | 创建或确认会话，验证请求中的知识库与文档，并在指定文档时反查所属知识库。 |
| 请求校验与运行状态 → 问题重写 | 初始化后的 `QueryGraphState` | 将本轮所有输入放入 LangGraph 状态，后续节点只读写这一状态。 |
| MongoDB → 问题重写 | 当前会话最近消息 | 为多轮指代消解和独立问题改写提供上下文。 |
| 问题重写 ↔ LLM | 历史、当前问题 ↔ 改写问题、主题列表、模式、深度与关注方面 | 把依赖上下文的问题改写为独立问题，并区分精确查询、讲解、总结、比较和澄清。 |
| 问题重写 → 原问题混合检索 | 改写问题、主题扩展词、知识库或文档范围 | 启动第一路本地检索；深度问题还会按关注方面生成补充查询并合并去重。 |
| 问题重写 → HyDE 混合检索 | 同一检索上下文 | 启动第二路召回，用假设性文档弥补用户问题和资料表达之间的差异。 |
| 问题重写 → Web Search | 改写问题 | 仅在 `enable_web_search=true` 时启动；失败时返回空列表，不中断本地链路。 |
| 问题重写 → 整份资料摘要 | 总结意图，或绑定文档的深度讲解意图 | 绕过 Top-K 问答检索，短文档直接全文综合，长文档进入 Map/Reduce。 |
| 原问题混合检索 ↔ Milvus | Dense/Sparse 查询向量 ↔ 候选切片 | 在 `kb_id` 或 `document_id` 范围内执行加权混合检索，返回第一份有序候选。 |
| HyDE 混合检索 ↔ LLM | 改写问题 ↔ 不含精确虚构事实的 HyDE 文本 | 生成更接近资料语言的检索扩展文本，不直接作为最终答案。 |
| HyDE 混合检索 ↔ Milvus | HyDE Dense/Sparse 向量 ↔ 候选切片 | 返回第二份有序候选。 |
| Web Search ↔ Web Search MCP | 搜索请求 ↔ 标题、正文摘要和 URL | 获得可选外部候选；它不进入本地 RRF。 |
| 两路本地检索 → RRF | 两份有序候选列表 | 按排名而非原始分数融合，降低不同检索分数尺度造成的偏差。 |
| RRF → Reranker | 去重后的本地融合候选 | 输出本地候选池并限制融合结果数量。 |
| Web Search → Reranker | 统一格式后的网页候选 | 在精排前与本地候选合并；近似重复时优先本地原文，因此网页结果不会挤占同内容的本地证据。 |
| Reranker ↔ BGE Reranker | 问题-证据文本对 ↔ 归一化相关性分数 | 重新排序全部候选，并依据阈值、分差和数量上限选择证据。 |
| Reranker → 证据约束回答 | `qualified`、`low`、`unscored` 或 `none` 证据 | 将证据质量显式交给回答节点；零召回时直接生成拒答状态。 |
| 整份资料摘要 ↔ Milvus | 知识库或文档范围 ↔ 最多 `SUMMARY_MAX_CHUNKS` 个切片 | 读取所选范围内的资料并按文档、章节和分片顺序排列。 |
| 整份资料摘要 ↔ LLM | 完整文档上下文，或分批 Map Prompt、Reduce Prompt ↔ 带引用综合结果 | 短文档一次生成，超出阈值的长文档分批处理，并保留范围、冲突和引用。 |
| 摘要结果 → 证据约束回答 | 已完成的摘要与来源 | 复用统一的保存和输出节点，不再执行常规问答生成。 |
| 证据约束回答 ↔ LLM | 编号证据、历史、问题、证据等级 ↔ 带 `[n]` 引用的答案 | 只允许依据候选正文回答，并要求精确值逐字来自证据。 |
| 证据约束回答 → MongoDB | 用户消息、助手答案、查询范围、模式、实际引用来源和图片 | 形成可继续追问、可回看和可审计的会话记录；短追问可继承最近来源的文档 ID。 |
| 证据约束回答 → 请求校验与运行状态 | 答案、来源、图片、任务轨迹 | 非流式请求形成完整 JSON；流式请求形成 `delta` 与 `final` 事件。 |
| 请求校验与运行状态 → 客户端 | JSON 或 SSE | 将最终结果和执行状态返回工作台或 API 调用方。 |

普通问答主链路：

```text
问题理解 -> 原问题/HyDE 并行召回 -> RRF -> Reranker -> 证据约束生成 -> 引用过滤
```

## 对话处理过程

一次对话同时使用两个标识：`session_id` 表示可持续多轮的会话，负责关联历史消息；`run_id` 表示单次请求，负责隔离任务状态、节点追踪和 SSE 事件。即使同一会话连续发起查询，每轮也有独立的 `run_id`。

```mermaid
sequenceDiagram
    participant U as 用户 / 工作台
    participant A as Query API
    participant G as LangGraph
    participant DB as MongoDB
    participant V as Milvus
    participant W as Web Search
    participant L as LLM
    participant R as BGE Reranker

    U->>A: POST /query
    A->>DB: 解析并校验知识库 / 文档范围，创建或确认会话
    A->>G: 初始化 session_id、run_id、范围与查询状态
    G->>DB: 读取最近对话历史
    G->>L: 消解指代、改写问题、生成查询计划
    par 原问题混合检索
        G->>V: Dense + Sparse
    and HyDE 混合检索
        G->>L: 生成假设性文档
        G->>V: Dense + Sparse
    and 可选联网检索
        G->>W: 搜索改写后的问题
    end
    G->>G: RRF 融合两路本地召回
    G->>R: 合并联网结果并提交问题-证据对
    R-->>G: 返回相关性分数
    G->>L: 按证据质量生成带引用答案
    G->>G: 过滤未引用来源与无效图片
    G->>DB: 保存助手消息、来源和图片
    G-->>A: 最终状态
    A-->>U: JSON 或 SSE final 事件
```

具体处理顺序如下：

1. **请求校验**：`POST /query` 校验问题非空，并按 `scope_mode` 解析范围。`all` 解析当前全部知识库，`knowledge_base` 校验 `kb_ids`，`documents` 校验 `document_ids` 并反查所属知识库；未知 ID 返回 `404`。未提供 `session_id` 时自动创建，随后为本轮生成新的 `run_id`。
2. **状态初始化**：API 建立任务状态，写入问题、知识库与文档范围、流式开关和默认关闭的联网开关，再调用查询图。流式请求立即返回 `run_id`，实际查询在后台执行。
3. **历史理解**：`node_item_name_confirm` 从 MongoDB 读取最近消息。LLM 结合历史消解指代，把当前问题改写成可独立检索的问题，并生成查询模式、深度和关注方面；“继续”“更详细一点”等短追问还会继承最近助手来源中的文档 ID。明确寒暄走本地短路回答，不加载向量模型。
4. **主题扩展与范围约束**：系统只在已解析的知识库或文档范围内检索。高置信主题用于增强召回，不作为文档硬过滤；指定文档时直接使用 `document_id in [...]`，不会召回同知识库的其他文档。
5. **全文综合路由**：整份总结，以及绑定具体资料的深度讲解，会绕过常规 Top-K 问答。正文总量不超过 `DIRECT_DOCUMENT_MAX_CHARS` 时一次读取完整上下文；更长时执行 Map/Reduce，并统一校验、压缩引用编号。
6. **并行召回**：普通问答同时执行原问题与 HyDE 的 Dense/Sparse 混合检索；深度问题按最多六个关注方面补充检索。Web Search 仅在 `enable_web_search=true` 时加入，失败只清空联网分支。
7. **融合、去重与精排**：RRF 先融合原问题和 HyDE 两路本地结果；联网结果随后统一结构。跨来源近似重复时优先本地原文，再由 BGE Reranker 重新打分和排序；深度与比较请求优先保留不同章节的证据。
8. **证据分级**：高于阈值的候选标记为 `qualified`；只有候选全部来自联网搜索且本地 Reranker 尚未就绪时，系统才按搜索顺序保留有限候选并标记为 `unscored`；有召回但全部低于阈值时保留少量候选并标记为 `low`，交给回答模型核验正文；所有分支均为零召回时直接拒答。
9. **受约束生成**：回答 Prompt 同时包含改写问题、历史、主题、候选正文、证据质量和可用图片。模型必须逐项核对正文，用 `[n]` 引用来源；数字、型号、日期等精确信息不得由常识补齐。
10. **输出后处理**：系统拒绝本轮不存在的引用，只保留答案中真实出现的来源，并把引用压缩为从 `[1]` 开始的连续编号；模型输出的图片 URL 必须存在于候选白名单，否则从答案中删除。
11. **持久化与返回**：用户消息和助手消息分别写入 MongoDB。非流式请求直接返回答案、来源、图片和已完成节点；流式请求通过 `/query/stream/{run_id}` 发送 `delta`，最后发送包含完整结果的 `final` 事件。
12. **失败隔离**：节点异常会把本轮任务标记为 `failed`，记录错误并在流式模式发送 `error` 事件。同一 `session_id` 下正在运行的请求不会因为删除会话而失去上下文，API 会返回 `409` 阻止删除。

主要分支行为：

| 场景 | 系统行为 |
|---|---|
| 你好、你是谁等明确寒暄 | 本地直接回答，跳过检索 |
| 明确要求整份资料摘要或深度讲解 | 短文档直接全文综合，长文档进入 Map/Reduce |
| 范围为全部知识库 | 后端解析当前全部有效知识库后检索 |
| 范围为指定文档 | 只检索所选 `document_id`，不扩大到同知识库 |
| 短追问未再次选择文档 | 从上一轮引用来源继承文档范围 |
| 显式范围为空且关闭联网 | 本地零扫描，最终拒答 |
| 显式范围为空且开启联网 | 只允许联网分支提供候选 |
| 联网搜索失败 | 本地链路继续，联网分支降级为空 |
| Reranker 未就绪且只有联网候选 | 按搜索顺序保留有限候选，由回答模型严格核验 |
| Reranker 未就绪且包含本地候选 | 本轮查询失败并记录错误，不静默跳过精排 |
| 已召回但得分低 | 传递少量低置信候选，不在排序节点武断拒答 |
| 全部召回为空 | 不调用证据生成链路，直接返回无资料提示 |

## 快速开始

### 环境要求

- Docker Desktop 与 Docker Compose
- NVIDIA GPU、驱动和 NVIDIA Container Toolkit（推荐）
- 可用的 OpenAI-compatible LLM API
- MinerU API Token（导入 PDF 时使用）
- 如需本地运行 Python：Python `>= 3.11` 与 `uv`

默认 Docker 配置启用 GPU。没有 CUDA 环境时，需要在 `.env.docker` 中将 `BGE_DEVICE` 和 `BGE_RERANKER_DEVICE` 改为 `cpu`，同时关闭 FP16，并移除或调整 `docker-compose.yml` 中的 GPU 配置。

### 1. 配置环境变量

```powershell
Copy-Item .env.docker.example .env.docker
```

至少修改以下占位值：

```dotenv
OPENAI_API_KEY=your-api-key
MINERU_API_TOKEN=your-mineru-token
MONGO_ROOT_PASSWORD=your-mongo-password
MINIO_ROOT_PASSWORD=your-minio-password
```

Compose 会根据服务账号自动生成应用连接配置，无需重复填写 MongoDB 和 MinIO 凭据。不要在公开仓库中提交真实密钥。

### 2. 启动服务

Windows 可直接运行：

```powershell
.\start.ps1
```

脚本会复用现有应用镜像；仅首次启动或镜像不存在时自动构建。修改 `pyproject.toml`、`uv.lock` 或 `Dockerfile` 后，使用以下命令重建：

```powershell
.\start.ps1 -Build
```

也可以使用 Docker Compose：

```powershell
# 首次构建
docker compose --env-file .env.docker up -d --build --remove-orphans

# 后续启动，无需重复构建
docker compose --env-file .env.docker up -d --remove-orphans
docker compose --env-file .env.docker ps
```

首次启动需要下载 BGE-M3 和 Reranker 模型，耗时取决于网络与磁盘速度。

### 3. 打开应用

| 服务 | 地址 |
|---|---|
| 知识库工作台 | <http://127.0.0.1:8001/query/html> |
| 导入 API 文档 | <http://127.0.0.1:8000/docs> |
| 查询 API 文档 | <http://127.0.0.1:8001/docs> |
| 导入服务健康检查 | <http://127.0.0.1:8000/health> |
| 查询服务健康检查 | <http://127.0.0.1:8001/health> |
| MinIO Console | <http://127.0.0.1:9101> |

查看日志：

```powershell
docker compose --env-file .env.docker logs -f app
```

停止服务：

```powershell
docker compose --env-file .env.docker down
```

## 配置说明

主要参数位于 `.env.docker`；本机直接运行时可参考 `.env.example`。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `LLM_DEFAULT_MODEL` | `qwen-flash` | OpenAI-compatible 对话模型 |
| `BGE_DEVICE` | `cuda:0` | Embedding 运行设备 |
| `BGE_RERANKER_DEVICE` | `cuda:0` | Reranker 运行设备 |
| `CHUNK_SIZE_TOKENS` | `500` | 文档切片目标大小 |
| `CHUNK_OVERLAP_TOKENS` | `80` | 普通切片重叠大小 |
| `DENSE_SPEC_GROUP_LINES` | `5` | 密集技术指标每组行数 |
| `RETRIEVAL_TOP_K` | `20` | 单路知识库召回数量 |
| `RRF_TOP_K` | `30` | RRF 输出上限 |
| `RERANK_MAX_TOP_K` | `8` | 最终证据上限 |
| `RERANK_MIN_TOP_K` | `2` | 合格证据的最低保留数量 |
| `RERANK_MIN_SCORE` | `0.09` | 重排最低相关度 |
| `RERANK_FALLBACK_TOP_K` | `3` | 无候选达到阈值时，交给回答模型复核的低置信候选数 |
| `MAX_UPLOAD_FILES` | `20` | 单次上传文件数上限 |
| `MAX_UPLOAD_SIZE_MB` | `50` | 单文件大小上限 |
| `CORS_ALLOW_ORIGINS` | 本地地址白名单 | 允许访问 API 的 Origin |

## 使用方式

### 创建知识库

```powershell
$kb = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/knowledge-bases" `
  -ContentType "application/json" `
  -Body '{"name":"设备说明书","description":"产品使用与维护资料"}'

$kb.kb_id
```

### 上传文档

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" `
  -F "kb_id=$($kb.kb_id)" `
  -F "files=@eval/datasets/documents/万用表RS-12的使用.md;type=text/markdown"
```

上传接口返回 `task_ids`。使用 `GET /status/{task_id}` 查询解析和向量导入进度。

### 查询知识库

```powershell
$body = @{
  query = "万用表使用的电池是什么型号？"
  session_id = "demo-session"
  scope_mode = "knowledge_base"
  kb_ids = @($kb.kb_id)
  document_ids = @()
  is_stream = $false
  enable_web_search = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8001/query" `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

示例回答：

```text
RS-12 数字万用表使用一粒 9V (NEDA 1604) 电池 [2]。
```

响应中的 `sources` 只包含答案实际引用的来源，可用于前端证据面板或后续审计。

`scope_mode` 支持 `all`、`knowledge_base` 和 `documents`。使用 `documents` 时传入 `document_ids`；联网补充默认关闭，只有明确需要外部资料时才设置 `enable_web_search = $true`。

## API 概览

### 导入服务 `:8000`

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 检查 MongoDB、Milvus、MinIO |
| `POST` | `/knowledge-bases` | 创建知识库 |
| `GET` | `/knowledge-bases` | 查询知识库列表 |
| `PATCH` | `/knowledge-bases/{kb_id}` | 修改知识库 |
| `DELETE` | `/knowledge-bases/{kb_id}` | 删除知识库及其数据 |
| `POST` | `/upload` | 上传并异步导入文档 |
| `GET` | `/status/{task_id}` | 查询导入状态 |
| `GET` | `/documents` | 查询全部文档，供查询范围选择 |
| `GET` | `/knowledge-bases/{kb_id}/documents` | 查询文档列表 |
| `DELETE` | `/documents/{document_id}` | 删除文档及关联数据 |
| `GET` | `/assets/{object_path}` | 代理访问私有 MinIO 资源 |

### 查询服务 `:8001`

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 检查 MongoDB、Milvus |
| `POST` | `/query` | 发起非流式或流式查询 |
| `GET` | `/query/stream/{run_id}` | 订阅 SSE 查询结果 |
| `GET` | `/status/{task_id}` | 查询任务状态与追踪信息 |
| `GET` | `/history/{session_id}` | 查询会话历史 |
| `GET/POST` | `/sessions` | 查询或创建会话 |
| `PATCH/DELETE` | `/sessions/{session_id}` | 修改或删除会话 |

完整请求与响应结构以 FastAPI 自动生成的 `/docs` 为准。

## Prompt 设计

`prompts/` 中的 10 个 Prompt 按职责拆分，并由回归测试校验占位符与关键约束：

| Prompt | 用途 |
|---|---|
| `rewritten_query_and_itemnames.prompt` | 多轮指代消解、主题抽取和检索问题改写；主观偏好问题转换为可由资料验证的选择依据 |
| `hyde_prompt.prompt` | 生成检索扩展文本，不直接回答问题，不虚构精确事实 |
| `answer_out.prompt` | 阅读候选正文、处理不同证据质量并生成逐项带 `[n]` 引用的答案 |
| `document_synthesis.prompt` | 在短文档完整上下文中直接完成总结或深度讲解，并保留逐项引用 |
| `summary_map.prompt` | 从长文档局部片段提取可引用事实，避免以局部代替全局 |
| `summary_reduce.prompt` | 合并分段摘要，保留冲突、范围、版本和引用 |
| `compress.prompt` | 在字符预算内压缩证据，同时保留数字、单位、否定、条件和对象关系 |
| `image_summary.prompt` | 以图片为主证据生成单行客观摘要，无法识别时明确返回固定提示 |
| `item_name_recognition.prompt` | 从标题和正文识别文档级规范主题名称 |
| `product_recognition_system.prompt` | 约束文档主题识别模型只返回单行名称或空字符串 |

所有用户输入、历史消息、检索来源、文档文本和图片上下文都按不可信数据处理。回答模型必须阅读正文，不能把重排分数当作事实判断：合格证据正常回答，低置信候选可以在正文确实支持时谨慎使用；只能部分回答时标明缺失部分，完全无依据时返回“当前资料中没有足够信息”。

## 评测结果

2026-08-18 使用仓库自带的两份原创合成 Markdown 评测夹具，并在自建集 `selfbuilt_zh_qa_v2` 上重新完成 14 条端到端评测：

| 指标 | 结果 |
|---|---:|
| 来源命中率 | **100%（12/12）** |
| 可回答准确率 | **100%（12/12）** |
| 不可回答拒答率 | **100%（2/2）** |
| 请求失败率 | **0%（0/14）** |
| 平均延迟 | **6.27 秒** |
| P50 / P95 / P99 | **5.14 / 11.91 / 19.96 秒** |

运行自建评测：

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py
```

复用已经完成导入的知识库：

```powershell
$env:EVAL_KB_ID = "<existing-kb-id>"
.\.venv\Scripts\python.exe eval\run_eval.py
```

结果保存在 `eval/results/result_selfbuilt_qa.json`。详细评测口径、官方数据集适配和历史结果边界见 [eval/README.md](eval/README.md)。

> 以上结果对应当前 `selfbuilt_zh_qa_v2` 的 14 条固定用例，评测运行 ID 为 `20260818T014526965891Z`；评测关闭联网并复用已完成导入的独立知识库。

## 测试

当前离线回归测试共 `55/55` 通过，其中包含共享 BGE-M3 模型并发编码、主体向量 FLOAT16 类型契约、三种查询范围、追问继承文档、深度方面扩展检索、跨来源去重、连续引用、默认关闭联网和 10 个 Prompt 渲染契约测试。

```powershell
$tests = @(
  "16_node_rerank.py",
  "17_text_compress_utils.py",
  "18_node_answer_output.py",
  "19_workspace_features.py",
  "20_rag_reliability.py",
  "21_reliability_hardening.py"
)

foreach ($test in $tests) {
  .\.venv\Scripts\python.exe (Join-Path "test" $test)
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

.\.venv\Scripts\python.exe -m compileall -q app eval test
git diff --check
```

测试覆盖重排截断与低置信回退、来源引用、知识库与文档范围、Prompt 契约、上传安全、会话状态、全文综合路由、文档删除和密集技术指标切分等关键行为。

## 安全与可靠性

- 上传文件名经过规范化，阻止路径穿越。
- 上传采用分块写入，并限制单文件大小和单次文件数量。
- CORS 使用显式白名单，不接受任意 Origin。
- 未知知识库 ID 返回 `404`，内部查询错误返回 `500`。
- MinIO Bucket 默认私有，通过受控资源接口访问。
- Prompt 将用户输入、历史、来源、文档和图片上下文统一标记为不可信数据，忽略角色覆盖、命令执行、伪造引用、输出协议覆盖和提示词泄露指令。
- 型号、编号、数字、日期和标准代号必须逐字来自引用证据。
- 资料只支持部分问题时回答可验证部分并标明缺口，不用模型常识补齐；冲突资料分别陈述，不强行合并。
- 重排阈值用于证据分级而非替回答模型作最终判断；低分但已召回的候选会被限制数量后交由回答模型逐条核验。
- 查询范围必须显式解析；指定文档时只使用 `document_id` 过滤，空范围不会隐式扫描本地资料。
- 联网补充在 API、状态默认值和工作台中均默认关闭。
- 评测使用独立运行会话，并对临时 `429/502/503/504` 执行有限重试。

## 项目结构

```text
PunditRAG/
├── app/
│   ├── import_process/       # 文档导入图与 :8000 API
│   ├── query_process/        # 查询图、工作台与 :8001 API
│   ├── clients/              # MongoDB、Milvus、MinIO 客户端
│   ├── conf/                 # 模型、检索和服务配置
│   ├── llm/                  # LLM、Embedding、Reranker 工具
│   └── utils/                # SSE、任务和通用工具
├── prompts/                  # 查询理解、HyDE、回答与摘要 Prompt
├── eval/                     # 自建及公开数据集评测
├── test/                     # 回归测试
├── eval/datasets/documents/  # 可再分发的原创合成评测夹具
├── docker-compose.yml        # 应用与依赖服务编排
├── Dockerfile
├── start.ps1                 # Windows 一键启动
└── README.md
```

## 参与贡献

欢迎提交 Issue 和 Pull Request。开发约定、测试命令和 PR 要求见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全问题请遵循 [SECURITY.md](SECURITY.md) 中的私密报告流程。版本变化记录在 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

本项目使用 [MIT License](LICENSE)。你可以使用、修改、分发和商用本项目，但必须在副本或主要部分中保留原版权声明和许可证文本。

第三方模型、数据集、文档和服务分别遵循其各自的许可证与使用条款，不因本项目采用 MIT License 而自动转为 MIT 授权。RGB、CRUD-RAG 和 MTRAG 的原始数据不会随仓库分发，详见 [eval/THIRD_PARTY_DATA.md](eval/THIRD_PARTY_DATA.md)。
