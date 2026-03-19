# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**El Paso** — a retrieval-as-a-service system that ingests organizational knowledge from Confluence and GitHub, then exposes it via MCP tools for any AI agent to search with source attribution. Consuming agents (Claude, Cursor, etc.) bring their own LLM — El Paso handles the search. There is also an optional `ask` endpoint that synthesizes answers via Qwen3 for lightweight consumers.

The target environment is a manufacturing-focused microservices org (Ping Golf) built on C#/.NET, RabbitMQ, PostgreSQL, and Blazor. The retrieval system itself is Python 3.10+ using Ollama (nomic-embed-text) for embeddings, Qdrant for vector storage, and Tree-sitter for C# code parsing.

## Commands

```bash
# Infrastructure (local dev — Qdrant only)
docker compose up -d qdrant   # Start Qdrant (port 6333, data persisted to qdrant_storage/)
ollama run qwen3              # Start/verify LLM

# Infrastructure (shared/production — Qdrant + MCP server)
docker compose up -d          # Start Qdrant + El Paso MCP server (port 8080)
docker compose logs -f elpaso # Watch MCP server logs

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

# MCP server (local stdio — for subprocess-based clients)
python mcp_server/server.py

# MCP server (local HTTP mode — for testing without Docker)
MCP_TRANSPORT=streamable-http python mcp_server/server.py

# MCP server (Docker — recommended for shared use)
# Included in `docker compose up -d`, endpoint: http://localhost:8080/mcp

# Tests
pytest                        # Run all tests
pytest tests/test_chunker.py  # Run a single test file
pytest -k "test_name"         # Run a specific test by name
```

## Architecture

Four layers with a clear data flow:

1. **Connectors** (`connectors/`) — Source-specific crawlers that return dataclasses (`ConfluencePage`, etc.) from external APIs (Confluence REST, GitHub via PyGithub). Four connectors: `confluence.py`, `github_docs.py`, `github_code.py`, `github_issues.py`
2. **Pipeline** (`pipeline/`) — Processes connector output: text chunking → embedding via Ollama → upsert into Qdrant
   - `store.py` (`VectorStore`): Qdrant collection management — `ensure_collection`, `upsert_chunks`, `delete_by_filter`, `scroll_by_filter`, `search`, `keyword_search`, `hybrid_search` (RRF merge) with optional source-type/repo/space filters. Creates payload indexes (full-text on `text`, keyword on `class_name`/`method_name`/`namespace`/`file_path`/`identifier`/`tags`) for hybrid search and community context management
   - `fingerprint.py`: Deterministic SHA-256 content fingerprinting for incremental ingestion (replaces Python's non-deterministic `hash()`)
   - `ingestion_tracker.py` (`IngestionTracker`): Persists `{fingerprint, ingested_at}` per source item keyed as `source_type::identifier` in `ingestion_state.json`. Enables incremental ingestion (skip unchanged content)
   - `embedder.py`: Wraps Ollama `/api/embed`. `vector_size()` probes the model for its embedding dimension
   - `logger.py`: Dual-output logging — human-readable to stderr, structured JSON to `logs/elpaso-YYYY-MM-DD.jsonl`
3. **Retrieval & Synthesis** (`mcp_server/retriever.py`) — Two-tier API:
   - `Retriever.search(question, scope, repo, space, top_k, mode)` — returns raw ranked chunks with scores + metadata (primary retrieval-as-a-service path). Includes TTL filtering for expired community context
   - `Retriever.ask(question, scope, repo, space)` — calls `search()` then synthesizes via Qwen3 (optional, for lightweight consumers that don't have their own LLM)
   - `Retriever.store(text, title, author?, tags?, expires_in_days?)` — stores community context: chunks, embeds, and upserts into Qdrant
   - `Retriever.delete(identifier)` — deletes community context by identifier
   - `Retriever.list_stored(tag?, limit?)` — lists stored community context grouped by identifier
   - Search modes: `"semantic"` (vector), `"keyword"` (text match), `"hybrid"` (RRF merge). Default: `"hybrid"` for code, `"semantic"` for others
   - `prompts.py`: System prompt (defines El Paso persona, citation rules) and `build_synthesis_prompt()` which formats retrieved chunks with source labels and URLs
4. **MCP Server** (`mcp_server/server.py`) — FastMCP wrapper exposing tools:
   - `ask_el_paso(question, scope?, repo?, space?)` — synthesized answer with citations
   - `search_el_paso(query, scope?, repo?, space?, top_k?, mode?)` — raw chunk retrieval
   - `search_code(query, repo?, top_k?, mode?)` — code-scoped search (hybrid by default)
   - `search_docs(query, space?, top_k?, mode?)` — docs-scoped search
   - `search_issues(query, repo?, top_k?, mode?)` — issues/PRs-scoped search
   - `store_context(text, title, author?, tags?, expires_in_days?)` — store community context (tribal knowledge)
   - `delete_context(identifier)` — delete stored community context
   - `list_context(tag?, limit?)` — list stored community context entries
   - Retriever is lazy-initialized on first call

### Chunking Strategy

- **Text docs** (`pipeline/chunker.py`): LlamaIndex sentence-window chunking (512 tokens, 50 overlap)
- **C# code** (`pipeline/csharp_chunker.py`): Tree-sitter AST parsing at class/method boundaries. Small classes → single chunk. Large classes → one chunk per method, each prefixed with context header (`// Namespace: X / // Class: Y : IY / // Method: Z`) and constructor. Interfaces always get one chunk
- **Code dispatch** (`pipeline/code_chunker.py`): Routes by language; currently only C# has a dedicated chunker, other languages fall back to whole-file

### Qdrant Payload Schema

Every point in the `el_paso` collection carries metadata used for filtering:
- `source_type`: `"confluence"`, `"github_docs"`, `"github_issue"`, `"github_pr"`, `"github_code"`, or `"community"`
- `text`: the chunk content
- Scope filtering maps: `"code"` → `github_code`, `"docs"` → `confluence + github_docs + community`, `"issues"` → `github_issue + github_pr`, `"confluence"` → `confluence`, `"all"` → no filter
- Payload indexes: full-text on `text` (word tokenizer), keyword on `class_name`, `method_name`, `namespace`, `file_path`, `identifier`, `tags`
- Code-specific: `repo_name`, `file_path`, `class_name`, `method_name`, `namespace`, `is_interface`, `implements_interfaces`
- Confluence-specific: `page_title`, `page_url`, `space_key`
- Community-specific: `identifier` (UUID grouping chunks from one submission), `title`, `author`, `tags`, `stored_at`, `expires_at` (optional TTL)

## Key Design Decisions

- Source attribution is mandatory — synthesis prompt enforces `[Source N]` citation for every claim, and instructs the LLM to refuse if context is insufficient
- Interface→implementation stitching: code chunks carry `implements_interfaces` metadata, enabling retrieval to connect SOLID-pattern C# interfaces with their implementations
- Collection must exist before processing: each ingestion script calls `store.ensure_collection()` at startup (before any `delete_by_filter` or `upsert_chunks`), since Qdrant returns 404 on operations against a nonexistent collection
- MCP server logs to stderr (not stdout) to avoid conflicts with MCP stdio transport
- MCP server supports two transports: `stdio` (default, for local subprocess clients) and `streamable-http` (for network access via Docker). Transport is controlled by `MCP_TRANSPORT` env var, falling back to `config.yaml` `server.transport`, falling back to `stdio`
- Qdrant data persists to `qdrant_storage/` via Docker volume mount
- In Docker, the MCP server reaches Qdrant via Docker DNS (`qdrant:6333`) and Ollama via `host.docker.internal:11434`

## Configuration

- `config.yaml`: Qdrant collection name, embedding/LLM model names, chunking params, retrieval top_k, search mode defaults (`search.default_mode`, `search.code_default_mode`, `search.rrf_k`), server transport/host/port (`server.transport`, `server.host`, `server.port`), GitHub repo prefix filter, file extension filters, skip patterns
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
