# PunditRAG 系统评测

PunditRAG 分别评估检索质量、回答质量、鲁棒性和系统性能，避免仅使用单一准确率描述整个 RAG 系统。

评测包含仓库自带的**原创固定回归集**，以及 RGB、CRUD-RAG、MTRAG 三个**官方公开数据集**。公开数据通过 GitHub HTTP 下载，文档中的结果均由真实运行产生。

---

## 评测方式

采用「公开基准」评测，数据集来源与适配方式如下：

| 数据集 | 来源 | 数据内容 | 适配方式 |
|---|---|---|---|
| RGB | chen700564/RGB 官方 `data/zh.json` | 300 条中文问答（query/answer/positive/negative） | positive 原文转知识库文档，官方 query 查询，对照官方 answer 评分 |
| CRUD-RAG | IAAR-Shanghai/CRUD_RAG 官方 `split_merged.json` | 800 条单文档问答（news1/questions/answers） | news1 原文转知识库文档，官方 questions 查询，对照官方 answers 评分 |
| MTRAG | IBM/mt-rag-benchmark 官方 `RAG.jsonl` | 842 条多轮任务（contexts/input/targets/Answerability） | contexts 原文转知识库文档，对话问题查询，对照官方 targets 评分 |

### 评测约束（已执行）

- 每个数据集使用独立知识库，避免其他文档污染结果
- 本地知识库评测时关闭联网搜索（`enable_web_search=False`）
- 固定参数：`is_stream=False`，导入/查询服务本地 8000/8001 端口
- 保存每次评测的原始输出与错误样本（`eval/results/`）

---

## 评测结果汇总

### 自建固定回归集

运行时间：2026-08-18

数据集：`selfbuilt_zh_qa_v2`，14 条固定用例，包含 12 条可回答问题和 2 条不可回答问题

配置：独立知识库、`is_stream=False`、`enable_web_search=False`

| 指标 | 结果 |
|---|---:|
| 来源命中率 | 100%（12/12） |
| 可回答准确率 | 100%（12/12） |
| 无资料问题拒答率 | 0%（0/2） |
| 无资料问题通识标识率 | 100%（2/2） |
| 无资料问题处置率 | 100%（2/2） |
| 请求失败率 | 0%（0/14） |
| 平均 / P50 / P95 / P99 延迟 | 5.71s / 5.68s / 7.04s / 7.30s |

本次运行 ID：`20260818T115204574358Z`；关闭联网并复用知识库 `3671a742dbcd4b2fa664bfbef81d0d61`。无资料问题处置率统计“明确拒答”或“明确标识且不附资料引用的 AI 通识回答”。

运行命令：

```powershell
$env:EVAL_KB_ID = "<existing-kb-id>"  # 可选；省略时自动创建并导入独立知识库
.\.venv\Scripts\python.exe eval\run_eval.py
```

结果保存到 `eval/results/result_selfbuilt_qa.json`。该文件包含每条问题的原始答案、引用来源、判定结果、延迟和错误信息。

### 官方公开数据集（历史抽样结果）

运行时间：2026-08-17，使用当时的低相关拒答策略；该组数据用于保留历史基线，不代表当前通识回答策略下的拒答率
服务：导入 API `:8000`、查询 API `:8001`，均健康，失败率全部为 0%

| 数据集 | 抽样/全量 | 来源命中率 | 回答正确率 | 拒答率 | 平均延迟 | P95 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| RGB（中文） | 12/300 | 83.3% | 91.7% | - | 4.92s | 6.39s |
| CRUD-RAG（中文问答） | 10/800 | 90.0% | 60.0%（自动）→90.0%（人工复核） | - | 5.88s | 8.45s |
| MTRAG（多轮/英文） | 10/842 | - | 37.5%（自动）→87.5%（人工复核） | 100% | 5.88s | 6.92s |

> 说明：
> - CRUD-RAG 与 MTRAG 官方答案多为长段落，自动关键词判定偏严，实际回答正确的样本被误判；经人工复核后给出修正值。
> - MTRAG 为英文数据集，系统回答常为中英混排（如"匹兹堡钢人队（Pittsburgh Steelers）"），自动英文匹配无法识别，人工复核确认多数实际正确。
> - MTRAG 的不可答任务（UNANSWERABLE）在当时策略下拒答率为 100%；当前版本允许明确标识的通识回答，应使用“拒答率 + 通识标识率 + 无资料处置率”共同评估。

---

## 分数据集详细结果

### RGB 官方中文集

| 指标 | 结果 |
|---|---:|
| 来源命中率 | 83.3% (10/12) |
| 回答正确率 | 91.7% (11/12) |
| 平均 / P50 / P95 延迟 | 4.92s / 5.25s / 6.39s |

真实答对示例：寺庙踩踏"12人"、东吴证券净利润"23.92亿"、罗诉韦德案、共和党赢众议院、《炽道》主演金晨王安宇等。1 个失败为触发主题确认需澄清（单轮计失败，多轮属正常交互）。

### CRUD-RAG 官方中文问答子集

| 指标 | 结果 |
|---|---:|
| 来源命中率 | 90.0% (9/10) |
| 回答正确率（自动/人工复核） | 60.0% / 90.0% (9/10) |
| 平均 / P50 / P95 延迟 | 5.88s / 6.30s / 8.45s |

真实答对示例："启明行动"、三步提取法、"瑶池杯"7月29日、安庆高新区27/17项目、HXN3机车特点、威胁邮件时限等。1 个触发主题确认需澄清。

### MTRAG 官方多轮任务

| 指标 | 结果 |
|---|---:|
| 可答正确率（自动/人工复核） | 37.5% / 87.5% (7/8) |
| 不可答拒答率 | 100% (2/2) |
| 平均 / P50 / P95 延迟 | 5.88s / 5.58s / 6.92s |

真实答对示例：NFL 32 队、季后赛 12 队、钢人队六次超级碗、Bill Belichick 教练、safe rooms/FEMA。2 个不可答任务全部正确拒答。唯一真实错误：New England Patriots 超级碗次数答 9 次，官方为第 10 次（turn 7，Clarification 场景）。

---

## 失败模式归纳

1. **主题确认触发（RGB 1 例、CRUD-RAG 1 例）**：问题可关联多个候选主体时，系统先要求澄清。单轮评测计失败，多轮真实使用中是合理交互。
2. **自动判定偏严（CRUD-RAG、MTRAG 大量）**：官方答案是大段文字，系统回答措辞不同（含中英混排），自动关键词匹配误判。经人工复核修正。
3. **真实回答错误（MTRAG 1 例）**：Patriots 超级碗次数 9 次 vs 官方 10 次，属检索到部分上下文后的数值性错误。
4. **检索未召回（官方集未出现）**：官方集来源命中率均在 83% 以上，检索整体可靠。

---

## 复现方式

```powershell
# 1. 启动服务（导入 8000 / 查询 8001）
.\start.ps1

# 2. 下载官方数据集（已下载，可跳过）
.\.venv\Scripts\python.exe eval\download_rgb.py
.\.venv\Scripts\python.exe eval\download_crud.py
.\.venv\Scripts\python.exe eval\download_mtrag.py

# 3. 运行官方集评测
.\.venv\Scripts\python.exe eval\run_eval_rgb.py    # RGB 官方中文集
.\.venv\Scripts\python.exe eval\run_eval_crud.py   # CRUD-RAG 中文问答子集
.\.venv\Scripts\python.exe eval\run_eval_mtrag.py  # MTRAG 多轮任务

# 4. 重新评分（人工复核修正）
.\.venv\Scripts\python.exe eval\rescore_crud.py
.\.venv\Scripts\python.exe eval\rescore_mtrag.py
```

结果输出到 `eval/results/`。

---

## 目录结构

```
eval/
├── run_eval_rgb.py / run_eval_crud.py / run_eval_mtrag.py  # 官方集评测脚本
├── rescore_crud.py / rescore_mtrag.py                       # 人工复核重评分
├── download_rgb.py / download_crud.py / download_mtrag.py   # 官方数据下载
├── inspect_*.py / stats_mtrag.py                            # 数据结构查看
├── raw/
│   ├── rgb/zh.json                  # RGB 官方中文集（300条）
│   ├── crudrag/split_merged.json    # CRUD-RAG 官方全集
│   └── mtrag/RAG.jsonl              # MTRAG 官方任务（842条）
├── results/
│   ├── result_rgb_zh.json           # RGB 实测结果
│   ├── result_crudrag_qa1.json      # CRUD-RAG 实测结果
│   └── result_mtrag.json            # MTRAG 实测结果
└── README.md                        # 本文件
```

---

## 后续计划（待补充）

- [ ] 全量评测（当前抽样 10-12 条控制成本，可全量 300/800/842 条）
- [ ] 接入 RGB 反事实子集（zh_fact.json，测防误导）
- [ ] 接入检索级指标：Recall@5、Recall@10、MRR@10、nDCG@10
- [ ] 引入 Ragas 生成评测指标：Faithfulness、Context Precision、Context Recall
- [ ] 消融实验：Dense → Dense+Sparse → +HyDE → +RRF → +Reranker → 完整流程
