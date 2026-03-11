# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**El Paso** — a locally-hosted RAG (Retrieval-Augmented Generation) system that ingests organizational knowledge from Confluence and GitHub, then exposes it via an MCP server for AI-powered Q&A with source attribution.

The target environment is a manufacturing-focused microservices org built on C#/.NET, RabbitMQ, PostgreSQL, and Blazor. The RAG system itself is Python.

## Current State

Phases 0–4 are **complete**. The full pipeline is operational: Confluence + GitHub docs/issues/PRs + C# source code ingested into Qdrant, with an MCP server exposing `ask_el_paso(question, scope?)` for AI-powered Q&A with source citations. Phase 5 (hardening/tuning) is next.

The roadmap lives in `docs/ElPaso_Roadmap.md`.

## Technology Stack

- **Python 3.11+**
- **Ollama** with Qwen3 (30B-A3B MoE) for LLM inference
- **Qdrant** (Docker) for vector storage
- **LlamaIndex** for RAG orchestration
- **Tree-sitter** for C#/Java/TypeScript/Python code parsing
- **Python MCP SDK** for the MCP server
- **Embedding**: nomic-embed-text via Ollama

## Architecture (Four Layers)

1. **Ingestion Layer** (`connectors/`) — Source-specific crawlers (Confluence, GitHub docs/issues/code) that extract raw content on a weekly schedule
2. **Processing Layer** (`pipeline/`) — Chunking (`chunker.py`, `code_chunker.py`, `csharp_chunker.py`), embedding (`embedder.py`), and vector storage (`store.py`)
3. **Retrieval & Synthesis Layer** — Query handling, semantic search against Qdrant, LLM-powered answer generation with source citations
4. **MCP Server Layer** (`mcp_server/`) — Exposes `ask_el_paso(question, scope?)` tool to AI hosts (Claude, Cursor) via Model Context Protocol

## Key Design Decisions

- Code chunking uses Tree-sitter at class/method boundaries with context headers (`// Namespace: X / // Class: Y : IY / // Method: Z`)
- Interface→implementation stitching via metadata for SOLID-pattern C# codebases
- Source attribution is mandatory on all answers — every claim must cite its source
- Synthesis prompts instruct LLM to answer only from retrieved context, never hallucinate
- Incremental ingestion tracks `last_modified` to avoid re-processing unchanged content

## Commands

```bash
# Infrastructure
docker compose up -d          # Start Qdrant
ollama run qwen3              # Start/verify LLM

# Dependencies
pip install -r requirements.txt

# Smoke test (validates Ollama + Qdrant + embeddings round-trip)
python smoke_test.py

# Ingestion (individual sources or all at once)
python scripts/ingest_confluence.py
python scripts/ingest_github_docs.py
python scripts/ingest_github_code.py
python scripts/run_weekly_ingest.py  # All sources

# Query test
python scripts/query_test.py

# MCP server
python mcp_server/server.py

# Tests (once test suite exists)
pytest                        # Run all tests
pytest tests/test_foo.py      # Run a single test file
pytest -k "test_name"         # Run a specific test by name
```

## Testing Strategy

- Use **pytest** as the test runner
- Prefer integration tests for pipeline and retrieval logic (test chunking→embedding→query round-trips)
- Unit test Tree-sitter parsing and chunking boundary logic
- Mock external services (Confluence, GitHub APIs) in tests; use real Qdrant via Docker for integration tests

## Configuration

- API credentials and endpoints go in `config.yaml` / `.env` (never commit secrets)
- Qdrant collection name: `el_paso` (defined in `config.yaml`)
- Target GitHub repos are filtered to 20 POC repositories

## Project Phases

The roadmap (`docs/ElPaso_Roadmap.md`) defines 6 phases:
- **Phase 0**: Foundation & infrastructure (Ollama, Qdrant, project skeleton)
- **Phase 1**: Confluence ingestion (end-to-end pipeline validation)
- **Phase 2**: GitHub docs & issues ingestion
- **Phase 3**: GitHub source code ingestion (Tree-sitter)
- **Phase 4**: MCP server with intelligent Q&A
- **Phase 5**: Hardening, tuning, automation
