<div align="center">

# PunditRAG

**An evidence-grounded RAG knowledge base for technical documents**

Document ingestion · Hybrid retrieval · Reranking · Streaming answers · Citation governance

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI Compatible](https://img.shields.io/badge/LLM-OpenAI--compatible-412991)](#configuration)

[中文](README.md) · [Quick start](#quick-start) · [Architecture](#architecture)

</div>

<p align="center"><img src="docs/assets/punditrag-hero.png" alt="PunditRAG overview" width="100%"></p>

PunditRAG is a self-hosted RAG system for papers, manuals, standards, product documents, and team knowledge bases. It combines dense and sparse retrieval, HyDE expansion, reciprocal-rank fusion, multilingual reranking, streaming generation, and citation validation in one practical pipeline.

## Highlights

- **Dense + sparse retrieval**: BGE-M3 captures semantic similarity and exact technical terms, identifiers, and abbreviations at the same time.
- **Multi-path recall**: the original question and a HyDE retrieval expansion run in parallel, reducing missed evidence caused by wording differences.
- **RRF fusion**: rank-based fusion avoids comparing incompatible scores from different retrieval channels.
- **Evidence-aware answering**: reranking, confidence tiers, citation whitelisting, invalid-citation rejection, and source deduplication.
- **Explicit scope**: `kb_ids` and `document_ids` constrain every local search; the system never silently scans every knowledge base.
- **OpenAI-compatible LLMs**: configure any compatible gateway such as OpenAI, DeepSeek, OrcaRouter, OpenRouter, SiliconFlow, or a local vLLM/Ollama endpoint.

## Architecture

<p align="center"><img src="docs/assets/punditrag-architecture.png" alt="PunditRAG architecture" width="100%"></p>

## Query workflow

<p align="center"><img src="docs/assets/punditrag-workflow.png" alt="PunditRAG query workflow" width="100%"></p>

## Quick start

```powershell
Copy-Item .env.docker.example .env.docker
.\start.ps1
```

Open <http://127.0.0.1:8001/query/html> after the services become healthy. See the Chinese README for full deployment, configuration, API, evaluation, and security details.

## Configuration

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.orcarouter.ai/v1
LLM_DEFAULT_MODEL=openai/gpt-4o-mini
```

The project uses local BGE-M3 embeddings and a local BGE reranker. LLM providers only affect generation, planning, HyDE, and optional vision tasks.

## Screenshots

![Workbench](docs/assets/punditrag-workbench.png)

![Execution trace](docs/assets/punditrag-trace.png)

## License

PunditRAG is released under the [MIT License](LICENSE).
