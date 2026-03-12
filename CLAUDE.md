# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**El Paso** — a locally-hosted RAG system that ingests organizational knowledge from Confluence and GitHub, then exposes it via an MCP server for AI-powered Q&A with source attribution.

The target environment is a manufacturing-focused microservices org built on C#/.NET, RabbitMQ, PostgreSQL, and Blazor. The RAG system itself is Python 3.11+ using Ollama (Qwen3 8B) for LLM inference, Qdrant for vector storage, and Tree-sitter for code parsing.

## Commands

```bash
# Infrastructure
docker compose up -d          # Start Qdrant (port 6333, data persisted to qdrant_storage/)
ollama run qwen3              # Start/verify LLM

# Environment setup
cp .env.example .env            # Then fill in API credentials
pip install -r requirements.txt # Uses .venv

# Smoke test (validates Ollama + Qdrant + embeddings round-trip)
python smoke_test.py

# Ingestion (unified — runs all sources)
python scripts/ingest_all.py
python scripts/ingest_all.py --source confluence   # Single source only
python scripts/ingest_all.py --source github_docs
python scripts/ingest_all.py --source github_code

# Ingestion (individual scripts — still work standalone)
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
2. **Pipeline** (`pipeline/`) — Processes connector output: text chunking → embedding via Ollama → upsert into Qdrant
   - `store.py` (`VectorStore`): Qdrant collection management — `ensure_collection`, `upsert_chunks`, `delete_by_filter`, `search`, `keyword_search`, `hybrid_search` (RRF merge) with optional source-type/repo/space filters. Creates payload indexes (full-text on `text`, keyword on `class_name`/`method_name`/`namespace`/`file_path`) for hybrid search
   - `fingerprint.py`: Deterministic SHA-256 content fingerprinting for incremental ingestion (replaces Python's non-deterministic `hash()`)
   - `ingestion_tracker.py` (`IngestionTracker`): Persists `{fingerprint, ingested_at}` per source item keyed as `source_type::identifier` in `ingestion_state.json`. Enables incremental ingestion (skip unchanged content)
   - `embedder.py`: Wraps Ollama `/api/embed`. `vector_size()` probes the model for its embedding dimension
   - `logger.py`: Dual-output logging — human-readable to stderr, structured JSON to `logs/elpaso-YYYY-MM-DD.jsonl`
3. **Retrieval & Synthesis** (`mcp_server/retriever.py`) — Two-tier API:
   - `Retriever.search(question, scope, repo, space, top_k, mode)` — returns raw ranked chunks with scores + metadata (retrieval-as-a-service)
   - `Retriever.ask(question, scope, repo, space)` — calls `search()` then synthesizes via Qwen3 (for lightweight consumers)
   - Search modes: `"semantic"` (vector), `"keyword"` (text match), `"hybrid"` (RRF merge). Default: `"hybrid"` for code, `"semantic"` for others
   - `prompts.py`: System prompt (defines El Paso persona, citation rules) and `build_synthesis_prompt()` which formats retrieved chunks with source labels and URLs
4. **MCP Server** (`mcp_server/server.py`) — FastMCP wrapper exposing tools:
   - `ask_el_paso(question, scope?, repo?, space?)` — synthesized answer with citations
   - `search_el_paso(query, scope?, repo?, space?, top_k?, mode?)` — raw chunk retrieval
   - `search_code(query, repo?, top_k?, mode?)` — code-scoped search (hybrid by default)
   - `search_docs(query, space?, top_k?, mode?)` — docs-scoped search
   - `search_issues(query, repo?, top_k?, mode?)` — issues/PRs-scoped search
   - Retriever is lazy-initialized on first call

### Chunking Strategy

- **Text docs** (`pipeline/chunker.py`): LlamaIndex sentence-window chunking (512 tokens, 50 overlap)
- **C# code** (`pipeline/csharp_chunker.py`): Tree-sitter AST parsing at class/method boundaries. Small classes → single chunk. Large classes → one chunk per method, each prefixed with context header (`// Namespace: X / // Class: Y : IY / // Method: Z`) and constructor. Interfaces always get one chunk
- **Code dispatch** (`pipeline/code_chunker.py`): Routes by language; currently only C# has a dedicated chunker, other languages fall back to whole-file

### Qdrant Payload Schema

Every point in the `el_paso` collection carries metadata used for filtering:
- `source_type`: `"confluence"`, `"github_docs"`, `"github_issue"`, `"github_pr"`, or `"github_code"`
- `text`: the chunk content
- Scope filtering maps: `"code"` → `github_code`, `"docs"` → `confluence + github_docs`, `"issues"` → `github_issue + github_pr`, `"confluence"` → `confluence`, `"all"` → no filter
- Payload indexes: full-text on `text` (word tokenizer), keyword on `class_name`, `method_name`, `namespace`, `file_path`
- Code-specific: `repo_name`, `file_path`, `class_name`, `method_name`, `namespace`, `is_interface`, `implements_interfaces`
- Confluence-specific: `page_title`, `page_url`, `space_key`

## Key Design Decisions

- Source attribution is mandatory — synthesis prompt enforces `[Source N]` citation for every claim, and instructs the LLM to refuse if context is insufficient
- Interface→implementation stitching: code chunks carry `implements_interfaces` metadata, enabling retrieval to connect SOLID-pattern C# interfaces with their implementations
- Collection must exist before processing: each ingestion script calls `store.ensure_collection()` at startup (before any `delete_by_filter` or `upsert_chunks`), since Qdrant returns 404 on operations against a nonexistent collection
- MCP server logs to stderr (not stdout) to avoid conflicts with MCP stdio transport
- Qdrant data persists to `qdrant_storage/` via Docker volume mount

## Configuration

- `config.yaml`: Qdrant collection name, embedding/LLM model names, chunking params, retrieval top_k, search mode defaults (`search.default_mode`, `search.code_default_mode`, `search.rrf_k`), GitHub repo prefix filter, file extension filters, skip patterns
- `.env` (from `.env.example`): `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN`, `GITHUB_TOKEN`, `GITHUB_ORG`, `QDRANT_HOST`, `QDRANT_PORT`, `OLLAMA_BASE_URL`

## Generated Artifacts (not committed)

- `qdrant_storage/` — Qdrant data (Docker volume mount)
- `.venv/` — Python virtual environment
- `ingestion_state.json` — Incremental ingestion tracker state
- `logs/` — Structured JSON logs (`elpaso-YYYY-MM-DD.jsonl`)

## Roadmap

See `docs/ElPaso_Roadmap.md`. Use `python scripts/ingest_all.py` for unified ingestion or individual scripts for single-source runs.

## Testing Strategy

- **pytest** as the test runner
- Prefer integration tests for pipeline/retrieval logic (chunking→embedding→query round-trips)
- Unit test Tree-sitter parsing and chunking boundary logic
- Mock external services (Confluence, GitHub APIs) in tests; use real Qdrant via Docker for integration tests
- Existing test files cover: chunker, csharp_chunker, confluence connector, github_docs connector, github_code connector, ingestion_tracker, retriever, fingerprint, hybrid_search, ingest_all
