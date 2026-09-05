<div align="center">

<h1>PunditRAG</h1>
<p><strong>A traceable RAG knowledge base for technical documents</strong></p>
<p>Import documents, ask questions, and follow citations back to the source.</p>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![CI](https://github.com/Pitkil/PunditRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Pitkil/PunditRAG/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-4e6b99)](LICENSE)

[简体中文](README.md) · **English**

[Features](#features) · [Screenshots](#screenshots) · [Quick start](#quick-start) · [Retrieval explained](#retrieval-explained) · [Development and evaluation](#development-and-evaluation)

</div>

<p align="center"><img src="docs/assets/punditrag-hero.png" alt="PunditRAG: from documents to cited answers" width="100%"></p>

PunditRAG brings document parsing, hybrid retrieval, and chat into one knowledge-base workbench. Import papers, manuals, standards, or product documents, select a knowledge base or individual documents, and inspect the source passages and execution trace behind each answer.

LangGraph orchestrates ingestion and queries. FastAPI provides the APIs and SSE streaming. Embedding and reranking run locally; generation uses a configurable OpenAI-compatible endpoint.

## Features

| Capability | What it provides |
|---|---|
| **Dense + sparse retrieval** | BGE-M3 combines semantic similarity with term matching for natural-language questions, identifiers, abbreviations, and terminology. |
| **Original query + HyDE** | Two retrieval branches search independently; RRF combines their ranked results to cover different ways of expressing the question. |
| **Multilingual reranking** | BGE Reranker jointly reads the question and candidate content to rank and select context. |
| **Document context** | Selected documents that fit the budget are read in full; retrieved anchors in longer selected documents receive same-section neighbors. |
| **Traceable answers** | Document-backed claims use `[n]` citations. Without evidence, the model may give an explicitly labeled general-knowledge answer. |
| **Interactive workbench** | Manage documents and sessions, inspect execution, delete individual messages, clear history, and optionally enable web search. |

Supported formats include PDF, Markdown, TXT, DOCX, PPTX, XLSX, CSV, HTML, and JSON. PDFs use the MinerU API; other supported formats are converted locally to Markdown before structured chunking.

## Screenshots

**Knowledge-base workbench** — select documents, ask questions, and inspect cited passages.

<p align="center"><img src="docs/assets/punditrag-workbench.png" alt="Actual workbench: scope selection, answers, and source passages" width="100%"></p>

<details>
<summary>View the execution trace</summary>

<p align="center"><img src="docs/assets/punditrag-trace.png" alt="Actual workbench execution trace" width="100%"></p>

</details>

## Quick start

### 1. Prerequisites

Install Docker Desktop / Docker Engine and Docker Compose, and obtain an API key for an OpenAI-compatible model service. PDF ingestion also requires a MinerU API token.

The default Compose setup requests an NVIDIA GPU. Linux hosts need NVIDIA Container Toolkit; Windows needs a GPU-capable Docker Desktop setup. For CPU use, set `BGE_DEVICE` and `BGE_RERANKER_DEVICE` to `cpu`, set both FP16 options to `0`, and remove `gpus: all` from Compose.

### 2. Clone and configure

```bash
git clone https://github.com/Pitkil/PunditRAG.git
cd PunditRAG
```

Windows PowerShell:

```powershell
Copy-Item .env.docker.example .env.docker
notepad .env.docker
```

On Linux / macOS, use `cp .env.docker.example .env.docker` and edit the file.

| Setting | Value to provide |
|---|---|
| `OPENAI_API_KEY` | Model service API key |
| `OPENAI_BASE_URL` | Compatible API base URL |
| `LLM_DEFAULT_MODEL` | Chat model identifier |
| `VL_MODEL` | An image-capable model available at the same endpoint |
| `MINERU_API_TOKEN` | PDF parsing token |
| `MONGO_ROOT_PASSWORD` / `MINIO_ROOT_PASSWORD` | Replace template passwords; Compose derives application connection settings |

The default template uses DashScope. Example text-model configuration for OrcaRouter:

```dotenv
OPENAI_API_KEY=replace-with-your-orcarouter-key
OPENAI_BASE_URL=https://api.orcarouter.ai/v1
LLM_DEFAULT_MODEL=openai/gpt-4o-mini
```

Use model identifiers from the provider's catalog and update `VL_MODEL` when switching providers. Chat, JSON output, and image input depend on the endpoint and selected model; configuration examples are not claims of live verification for every provider. Optional provider parameters can be supplied as a JSON object in `OPENAI_EXTRA_BODY_JSON`.

### 3. Start services

Windows:

```powershell
.\start.ps1
```

The script builds the application image when missing and reuses it on subsequent starts. Use `.\start.ps1 -Build` to rebuild. Initial use also downloads the local embedding and reranking models.

Alternatively:

```bash
docker compose --env-file .env.docker up -d --build
```

Omit `--build` on later starts when no rebuild is needed.

### 4. Ask your first question

Open the [workbench](http://127.0.0.1:8001/query/html), create a knowledge base, upload documents, and wait for ingestion to finish. Select the knowledge base or individual documents and ask a question.

Try the included [sample manual](eval/datasets/documents/万用表RS-12的使用.md) and ask which battery the multimeter uses. Inspect the cited passage to verify the answer.

<details>
<summary>Service addresses and operations</summary>

| Service | Address |
|---|---|
| Workbench | http://127.0.0.1:8001/query/html |
| Ingestion API docs | http://127.0.0.1:8000/docs |
| Query API docs | http://127.0.0.1:8001/docs |
| MinIO Console | http://127.0.0.1:9101 |

```bash
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs -f app
docker compose --env-file .env.docker down
```

</details>

## Retrieval explained

### Dense and sparse: complementary signals

**Dense retrieval** represents questions and passages as continuous vectors and searches for semantic similarity. A question about battery replacement intervals may retrieve a passage about operating time even when the wording differs.

**Sparse retrieval** uses BGE-M3's learned token weights to contribute term-matching signals for names, abbreviations, and technical identifiers. This implementation uses learned sparse representations, not a separate BM25 integration. Exact identifiers still need to be checked against the source.

Both searches use inner product in Milvus. Within each passage-search branch, `WeightedRanker` combines dense and sparse results with default weights of `0.5 / 0.5`.

### Original query and HyDE: two search expressions

The normal retrieval path runs two branches, each using Dense + Sparse:

| Branch | Input | Purpose |
|---|---|---|
| Original query | The user's question with selected document names or matched topics as search context | Preserve the actual question and explicit scope. |
| HyDE | A model-generated hypothetical document | Search using passage-like wording. The generated text is a search aid, never answer evidence. |

Matched `item_name` topics also trigger supplemental searches within scope. Their results are merged and deduplicated with the broad scoped search. Topic matching does not replace knowledge-base or document filters.

HyDE has a default 10-second request timeout. On failure it returns no candidates, allowing the original-query branch to continue.

### RRF: combine branch rankings

After weighted dense/sparse fusion within each branch, `node_rrf` merges the original-query and HyDE lists:

```text
RRF(d) = Σ 1 / (60 + rank_i(d))
```

Both branches have weight 1. A passage appearing in multiple lists accumulates contributions. RRF combines ranks instead of adding raw similarity scores from the branches; the default output limit is 30 candidates.

When enabled, web results join local candidates at the reranker. They are not included in this local RRF calculation.

### Reranking: select answer context

BGE Reranker jointly processes the original question, document title, section, and passage text. Candidates are then deduplicated, assigned confidence tiers, and truncated using relevance thresholds and score gaps.

If every candidate falls below the threshold, a bounded set of nonzero low-scoring passages is retained for the answer model to inspect. Relevance scores select context; they are not truth scores or answer-accuracy probabilities.

For explicitly selected documents, final local anchors receive neighboring passages from the same document and section, defaulting to `part ± 1`. Expansion takes place after reranking.

## Architecture and conversation flow

<p align="center"><img src="docs/assets/punditrag-architecture.png" alt="Lifecycle: import, chunk, embed, retrieve, rerank, answer" width="100%"></p>

Ingestion converts content to Markdown, splits it by headings, paragraphs, and tables, and attaches document topics and metadata. BGE-M3 embeddings go to Milvus; MongoDB stores document and conversation records, while MinIO stores images processed through the Markdown image path.

<p align="center"><img src="docs/assets/punditrag-workflow.png" alt="Normal retrieval path: question, scope, parallel recall, fusion, reranking, streaming answer" width="100%"></p>

The figure illustrates normal retrieval. The full workflow also includes direct document reading, full-document synthesis, and optional web search:

| Handoff | Behavior |
|---|---|
| User → API | Validate question and scope, associate the session, and assign a per-request `run_id`. |
| API → Planning | Read recent session history, extract recall topics, and decide whether full-document synthesis is needed. Preserve the original question. |
| Planning → Document context | Read complete selected documents when ingestion and chunk counts are complete and the combined text fits the budget; this route requires web search and full-document synthesis to be off. |
| Retrieval → RRF | Run original-query and HyDE hybrid search, then fuse local ranked lists. |
| RRF / optional web → Reranker | Merge, rerank, deduplicate, and add neighbors for selected documents. |
| Full-document synthesis → Answer | Use short-document synthesis or long-document Map/Reduce with source associations. |
| Context → Answer model | Supply the current question, recent history, and passages. Cite document-backed claims; allow explicitly labeled general knowledge without invented sources. |
| Answer → User / MongoDB | Validate citations and images, assemble sources, and save messages. Streaming sends `delta` events followed by a complete `final` result. |

`session_id` links conversation history; `run_id` separates request execution and SSE events. Answer text starts streaming at generation; earlier retrieval progress is visible in the workbench.

<details>
<summary>Retrieval and context settings</summary>

Defaults are defined in [retrieval_config.py](app/conf/retrieval_config.py).

| Setting | Default | Meaning |
|---|---:|---|
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | 500 / 80 | Approximate token budget and overlap |
| `RETRIEVAL_TOP_K` | 20 | Broad hybrid-search result limit |
| `TOPIC_EXPANSION_TOP_K` | 10 | Supplemental topic-search limit |
| `RRF_TOP_K` / `RERANK_INPUT_TOP_K` | 30 / 30 | Fused output and local reranking input limits |
| `RERANK_MAX_TOP_K` | 8 | Evidence limit before neighbor expansion |
| `RERANK_MIN_TOP_K` | 2 | Minimum retention target during gap truncation; does not fabricate candidates |
| `RERANK_MIN_SCORE` | 0.09 | Relevance tier threshold |
| `RERANK_FALLBACK_TOP_K` | 8 | Low-confidence fallback limit |
| `DIRECT_DOCUMENT_MAX_CHARS` | 64000 | Combined selected-document chunk text budget in characters, not the model token limit |
| `NEIGHBOR_EXPAND_PARTS` | 1 | Same-section neighbors on either side of an anchor |

</details>

## API and code guide

Ingestion runs on `:8000` and queries on `:8001`. See each service's FastAPI `/docs` for complete schemas.

| Endpoint | Purpose |
|---|---|
| `POST :8000/knowledge-bases` | Create a knowledge base |
| `POST :8000/upload` | Upload documents and receive task identifiers |
| `GET :8000/status/{task_id}` | Inspect ingestion progress |
| `POST :8001/query` | Start a streaming or non-streaming query |
| `GET :8001/query/stream/{run_id}` | Subscribe to SSE |
| `GET :8001/history/{session_id}` | Read history |
| `DELETE :8001/history/{session_id}/messages/{message_id}` | Delete one message |
| `DELETE :8001/history/{session_id}` | Clear session history |

Suggested reading order:

1. [Ingestion graph](app/import_process/agent/main_graph.py): format routing, parsing, chunking, topics, and vector storage.
2. [Query graph](app/query_process/agent/main_graph.py): document context, parallel recall, and generation.
3. [Retrieval helpers](app/query_process/agent/retrieval_utils.py), [RRF](app/query_process/agent/nodes/node_rrf.py), and [reranker](app/query_process/agent/nodes/node_rerank.py): filters, fusion, and neighbor expansion.
4. [Answer node](app/query_process/agent/nodes/node_answer_output.py) and [prompts](prompts/): questions, history, evidence, and citations.
5. [Query API](app/query_process/api/server.py): workbench, sessions, and SSE.

## Development and evaluation

Local development uses Python 3.11+ and `uv`. See [.env.example](.env.example) for local configuration.

```bash
uv sync --frozen
uv run --no-sync python test/16_node_rerank.py
uv run --no-sync python test/18_node_answer_output.py
uv run --no-sync python test/19_workspace_features.py
uv run --no-sync python test/20_rag_reliability.py
uv run --no-sync python test/21_reliability_hardening.py
```

[GitHub Actions](.github/workflows/ci.yml) defines offline regression, Python compilation, and Dockerfile checks. Main-branch pushes and manual runs also build the image. Check [Actions](https://github.com/Pitkil/PunditRAG/actions) for current results.

Evaluation includes a [fixed local dataset](eval/run_eval.py) and adapters for RGB, CRUD-RAG, and MTRAG. With services and model credentials configured:

```bash
uv run --no-sync python eval/run_eval.py
```

See the [evaluation guide](eval/README.md) for data sources, commands, and historical sample results. Public-dataset scores recorded on August 17, 2026 apply to that version and sample; this README update did not rerun evaluation.

## Deployment notes

Embedding and reranking run locally. Questions, candidate passages, and relevant history are sent to the configured LLM service. PDFs are sent to MinerU, and image summaries use the configured vision model. Optional web search is off by default and currently uses a separately configured DashScope Web Search MCP; changing the LLM Base URL does not change this service.

Use the workbench locally or on a trusted network. Configure authentication and access control before exposing it externally; see [SECURITY.md](SECURITY.md).

## Contributing and license

Share feedback through [Issues](https://github.com/Pitkil/PunditRAG/issues), or contribute documentation, tests, and features. See the [contribution guide](CONTRIBUTING.md) and [changelog](CHANGELOG.md).

PunditRAG uses the [MIT License](LICENSE). Third-party models, services, and datasets retain their own licenses; see [third-party data notes](eval/THIRD_PARTY_DATA.md).
