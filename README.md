# El Paso

A locally-hosted RAG (Retrieval-Augmented Generation) system that ingests organizational knowledge from Confluence and GitHub, then exposes it via an [MCP](https://modelcontextprotocol.io/) server for AI-powered Q&A with source attribution.

Built for a manufacturing-focused microservices environment (C#/.NET, RabbitMQ, PostgreSQL, Blazor). The RAG system itself is Python.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  Ingestion Layer│────▶│ Processing Layer  │────▶│ Retrieval & Synth.  │────▶│  MCP Server  │
│  (connectors/)  │     │  (pipeline/)      │     │                     │     │(mcp_server/) │
│                 │     │                   │     │                     │     │              │
│ Confluence      │     │ Chunking          │     │ Semantic search     │     │ ask_el_paso()│
│ GitHub docs     │     │ Embedding (Ollama)│     │ LLM synthesis       │     │              │
│ GitHub code     │     │ Qdrant storage    │     │ Source attribution   │     │              │
└─────────────────┘     └──────────────────┘     └─────────────────────┘     └──────────────┘
```

## Tech Stack

- **Python 3.11+**
- **Ollama** with Qwen3 8B for LLM inference
- **Qdrant** (Docker) for vector storage
- **LlamaIndex** for RAG orchestration
- **Tree-sitter** for C#/Java/TypeScript/Python code parsing
- **Python MCP SDK** for the MCP server

## Quick Start

```bash
# Start infrastructure
docker compose up -d          # Qdrant
ollama run qwen3              # LLM

# Install dependencies
pip install -r requirements.txt

# Validate the stack
python smoke_test.py
```

## Ingestion

```bash
python scripts/ingest_confluence.py     # Confluence pages
python scripts/ingest_github_docs.py    # GitHub READMEs, issues, PRs
python scripts/ingest_github_code.py    # GitHub source code (Tree-sitter)
python scripts/rebuild_collection.py   # Drop & re-ingest all sources
```

## MCP Server

```bash
python mcp_server/server.py
```

Exposes `ask_el_paso(question, scope?)` to any MCP-compatible AI host (Claude, Cursor, etc.). Every answer includes source citations.

## Roadmap

See [`docs/ElPaso_Roadmap.md`](docs/ElPaso_Roadmap.md) for the full roadmap. Phases 0–4 complete, currently in **Phase 5**.

| Phase | Title |
|-------|-------|
| 0 | Foundation & Infrastructure |
| 1 | Confluence Ingestion |
| 2 | GitHub Docs & Issues |
| 3 | GitHub Source Code (Tree-sitter) |
| 4 | MCP Server / Intelligent Q&A |
| 5 | Hardening & Optimization |

## Configuration

Copy `.env.example` to `.env` and fill in your API credentials for Confluence and GitHub. See `config.yaml` for additional settings.

## Testing

```bash
pytest                        # All tests
pytest tests/test_foo.py      # Single file
pytest -k "test_name"         # By name
```

## License

Private / Internal Use
