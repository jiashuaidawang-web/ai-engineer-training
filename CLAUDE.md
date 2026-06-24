# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the "AI Engineering Training Camp" (AI工程化训练营) monorepo — a 11-week curriculum covering the full stack of AI application engineering. Each `weekNN/` directory is an independent learning module with its own `pyproject.toml`, `.env`, and virtual environment. Homework examples are collected under `homework_examples/`.

## Common Development Commands

All weeks use **Python 3.11+** and **[uv](https://github.com/astral-sh/uv)** for dependency management.

```bash
# In any weekNN/ directory:
uv sync --locked                          # Install dependencies from pyproject.toml + uv.lock
source .venv/bin/activate                 # Activate virtual environment (Windows: .venv\Scripts\activate)

# Week-specific notes:
# week02: Has torch dependency; may need `uv pip compile pyproject.toml --output-file requirements.txt`
#          If torch wheel mismatch on Intel Mac, pin to torch==2.2.2 in pyproject.toml
# week03/week04/week05/week06: Jupyter notebooks — register kernel with:
#          python -m ipykernel install --user --name=weekXX --display-name="AI工程化(weekXX)"
#          Then run `jupyter lab` and select the kernel
```

### Running Apps

Each week has its own entry points. Key ones:

| Week | Purpose | Command |
|------|---------|---------|
| week02 | Local fine-tuning platform (FastAPI + Gradio) | `python -m local_ft.server` → http://localhost:7866 |
| week03-local-rag | Local RAG system (FastAPI + Gradio + LlamaIndex + FAISS) | `uvicorn main:app --host 0.0.0.0 --port 7866` |
| week03 | LlamaIndex notebooks | `jupyter lab` |
| week04 | LangChain notebooks | `jupyter lab` |
| week05 | Multi-Agent collaboration (LangGraph) | `jupyter lab` |
| week06 | DSL language design | `jupyter lab` |
| week07 | Advanced Agent capabilities | `jupyter lab` |
| week08 | LLM serving, K8s, monitoring, Ray, ELK | See `week08/README.md` for per-module commands |
| week09 | Python high-performance concurrency | Various `pXX_*.py` scripts in `week09/` |
| week10 | Full customer service agent platform | FastAPI on port 8000 (see API docs in week10/README.md) |
| week11-homework | Werewolf game multi-agent system | See `week11-homework/README.md` |

### Environment Variables

Most weeks require API keys. Copy `.env.example` → `.env` in the respective week directory:
- `OPENAI_API_KEY` — OpenAI-compatible API (used across many weeks)
- `DASHSCOPE_API_KEY` — Alibaba DashScope / Qwen models (week03, week08, week09)
- Additional env vars documented per-week README

## Repository Structure

```
├── README.md                      # Top-level index of all weeks
├── week01/                        # LLM API basics: OpenAI client, LangChain, LangGraph, AutoGen, LlamaIndex
├── week02/                        # Model fine-tuning: LoRA, local FT platform (FastAPI+Gradio)
├── week03/                        # LlamaIndex notebooks (RAG fundamentals)
├── week03-local-rag/              # Complete local RAG system (upload → index → chat)
├── week03-qanything/              # QAnything project: full RAG with Docker compose, two-stage retrieval
├── week04/                        # LangChain notebooks
├── week05/                        # Multi-Agent collaboration & communication (LangGraph)
├── week06/                        # DSL language design & execution engine
├── week07/                        # Advanced Agent capabilities
├── week08/                        # LLM service deployment: FastAPI, Docker, K8s, Ray, ELK, Prometheus
├── week09/                        # Python concurrent programming: asyncio, multiprocessing, profiling
├── week10/                        # Production customer service agent platform (full-stack)
├── week11-homework/               # Final project: Werewolf multi-agent game
├── homework_examples/             # Reference solutions from students
│   ├── week03-homework/           # Chunking research, OCR research
│   ├── week03-homework-2/         # Graph RAG, Milvus FAQ indexing (FastAPI + docker-compose)
│   ├── week04-homework/           # Smart customer service with LangGraph tools
│   ├── week05-homework/           # Multi-agent LangGraph application
└── .env / .env.example            # API key templates (per-week)
```

## Architecture Patterns by Week

- **week01**: Foundation — OpenAI API calls, LangChain chains, LangGraph state machines, AutoGen multi-agent, LlamaIndex RAG, model routing/deep thinking
- **week02**: Fine-tuning pipeline — data preparation, LoRA/QLoRA training with `ms-swift`, Gradio UI for upload/quantization/model merging
- **week03**: RAG — LlamaIndex document ingestion, FAISS vector stores, two-stage retrieval (vector search + DashScope reranker)
- **week03-local-rag**: Production-grade RAG — FastAPI backend, Gradio frontend, multi-format file parsing (PDF/DOCX/XLSX/TXT/CSV), persistent knowledge bases
- **week03-qanything**: Enterprise RAG — Docker compose stack, Milvus vector DB, BCEmbedding for bilingual embedding, two-stage retrieval, offline-capable
- **week04**: LangChain — chains, agents, tool calling, memory
- **week05**: Multi-Agent — LangGraph state machines, agent collaboration patterns
- **week06**: DSL — custom language design and execution engine
- **week07**: Advanced Agents — tool use, planning, memory
- **week08**: DevOps for AI — FastAPI service wrapping, Docker containerization, K8s deployments (baseline/canary/Istio), Ray Serve, ELK logging, Prometheus metrics
- **week09**: Concurrency — asyncio event loop, Future/Task, multiprocessing, GIL analysis, profiling (cProfile, py-spy), async FastAPI with WebSocket/SSE
- **week10**: Full platform — multi-tenant customer service agent with RAG, tool calling (order queries, human transfer), MCP protocol, multimodal input, model switching, health checks, suggestion engine
- **week11-homework**: Final project — werewolf game with moderator agent, role-playing agents, episodic + semantic memory (FAISS/Milvus), RAG-enhanced reasoning

## Key Dependencies

- **LLM APIs**: `openai`, `dashscope` (Alibaba Cloud)
- **Frameworks**: `langchain`, `langgraph`, `llama-index`, `autogen-agentchat`
- **Fine-tuning**: `torch`, `ms-swift` (ModelScope), `peft`/`transformers`
- **RAG**: `faiss-cpu`, `milvus`, `BCEmbedding` (in week03-qanything)
- **Serving**: `fastapi`, `uvicorn`, `gradio`, `ray[serve]`
- **Infra**: `docker-compose`, Kubernetes YAMLs, `prometheus-client`, ELK stack
- **Concurrency**: `aiohttp`, `asyncpg`, `redis`, `httpx`, `celery`
- **Testing**: `pytest`

## Working with This Repo

1. **Navigate to the week directory** you want to work with — each week is self-contained with its own `pyproject.toml`, `.venv`, and dependencies.
2. **Check the week's README.md** for specific setup instructions, kernel registration (for notebooks), and run commands.
3. **Homework examples** in `homework_examples/` show reference implementations for weeks 03–05.
