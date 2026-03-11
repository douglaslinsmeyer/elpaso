# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**El Paso** — a locally-hosted RAG system that ingests organizational knowledge from Confluence and GitHub, then exposes it via an MCP server for AI-powered Q&A with source attribution.

The target environment is a manufacturing-focused microservices org built on C#/.NET, RabbitMQ, PostgreSQL, and Blazor. The RAG system itself is Python 3.11+ using Ollama (Qwen3 8B) for LLM inference, Qdrant for vector storage, and Tree-sitter for code parsing.

## Commands

```bash
# Infrastructure
docker compose up -d          # Start Qdrant (port 6333)
ollama run qwen3              # Start/verify LLM

# Dependencies
pip install -r requirements.txt

# Smoke test (validates Ollama + Qdrant + embeddings round-trip)
python smoke_test.py

# Ingestion (individual sources)
python scripts/ingest_confluence.py
python scripts/ingest_github_docs.py
python scripts/ingest_github_code.py

# Full rebuild (drops collection, clears state, re-ingests all)
python scripts/rebuild_collection.py

# Query test (interactive retrieval check)
python scripts/query_test.py

# MCP server
python mcp_server/server.py

# Tests
pytest                        # Run all tests
pytest tests/test_chunker.py  # Run a single test file
pytest -k "test_name"         # Run a specific test by name
```

## Architecture

Four layers with a clear data flow:

1. **Connectors** (`connectors/`) — Source-specific crawlers that return dataclasses (`ConfluencePage`, etc.) from external APIs (Confluence REST, GitHub via PyGithub)
2. **Pipeline** (`pipeline/`) — Processes connector output: text chunking → embedding via Ollama → upsert into Qdrant. The `IngestionTracker` (`ingestion_state.json`) stores fingerprints to enable incremental ingestion (skip unchanged content)
3. **Retrieval & Synthesis** (`mcp_server/retriever.py`) — Embeds the user's question → searches Qdrant with optional filters → deduplicates results → sends context + question to Qwen3 via Ollama chat API → returns answer with `[Source N]` citations
4. **MCP Server** (`mcp_server/server.py`) — FastMCP wrapper exposing `ask_el_paso(question, scope?, repo?, space?)`. Retriever is lazy-initialized on first call

### Chunking Strategy

- **Text docs** (`pipeline/chunker.py`): LlamaIndex sentence-window chunking (512 tokens, 50 overlap)
- **C# code** (`pipeline/csharp_chunker.py`): Tree-sitter AST parsing at class/method boundaries. Small classes → single chunk. Large classes → one chunk per method, each prefixed with context header (`// Namespace: X / // Class: Y : IY / // Method: Z`) and constructor. Interfaces always get one chunk
- **Code dispatch** (`pipeline/code_chunker.py`): Routes by language; currently only C# has a dedicated chunker, other languages fall back to whole-file
- **Embedder** (`pipeline/embedder.py`): Wraps Ollama `/api/embed`. `vector_size()` probes the model for its embedding dimension, used at ingestion startup to ensure the Qdrant collection exists before any deletes or upserts

### Qdrant Payload Schema

Every point in the `el_paso` collection carries metadata used for filtering:
- `source_type`: `"confluence"`, `"github_docs"`, `"github_issue"`, `"github_pr"`, or `"github_code"`
- `text`: the chunk content
- Scope filtering maps: `"code"` → `github_code`, `"docs"` → `confluence + github_docs/issue/pr`, `"all"` → no filter
- Code-specific: `repo_name`, `file_path`, `class_name`, `method_name`, `namespace`, `is_interface`, `implements_interfaces`
- Confluence-specific: `page_title`, `page_url`, `space_key`

## Key Design Decisions

- Source attribution is mandatory — synthesis prompt enforces `[Source N]` citation for every claim, and instructs the LLM to refuse if context is insufficient
- Interface→implementation stitching: code chunks carry `implements_interfaces` metadata, enabling retrieval to connect SOLID-pattern C# interfaces with their implementations
- Incremental ingestion: `IngestionTracker` persists `{fingerprint, ingested_at}` per source item keyed as `source_type::identifier`. On re-run, only changed items are re-processed
- Collection must exist before processing: each ingestion script calls `store.ensure_collection()` at startup (before any `delete_by_filter` or `upsert_chunks`), since Qdrant returns 404 on operations against a nonexistent collection
- MCP server logs to stderr (not stdout) to avoid conflicts with MCP stdio transport
- Structured JSON logging to `logs/elpaso-YYYY-MM-DD.jsonl` for production debugging

## Configuration

- `config.yaml`: Qdrant collection name, embedding/LLM model names, chunking params, retrieval top_k, GitHub repo prefix filter, file extension filters, skip patterns
- `.env` (from `.env.example`): `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN`, `GITHUB_TOKEN`, `GITHUB_ORG`, `QDRANT_HOST`, `QDRANT_PORT`, `OLLAMA_BASE_URL`

## Current State

Phases 0–4 are **complete**. Phase 5 (hardening/tuning) is next. Roadmap: `docs/ElPaso_Roadmap.md`.

Note: `scripts/run_weekly_ingest.py` is referenced in docs but does not exist yet — use individual ingest scripts or `rebuild_collection.py` for now.

## Testing Strategy

- **pytest** as the test runner
- Prefer integration tests for pipeline/retrieval logic (chunking→embedding→query round-trips)
- Unit test Tree-sitter parsing and chunking boundary logic
- Mock external services (Confluence, GitHub APIs) in tests; use real Qdrant via Docker for integration tests
- Existing test files cover: chunker, csharp_chunker, confluence connector, github_docs connector, github_code connector, ingestion_tracker, retriever
