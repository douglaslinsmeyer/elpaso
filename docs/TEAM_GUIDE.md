# El Paso — Team Guide

## What is El Paso?

El Paso is a RAG (Retrieval-Augmented Generation) system that indexes your team's knowledge from Confluence, GitHub repos, and source code, then lets you query it through AI tools with source-cited answers.

## Connecting to El Paso

### Claude Desktop

Add to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "el-paso": {
      "command": "wsl",
      "args": ["-e", "bash", "-c", "cd /home/brianba/projects/sandbox/elpaso && source .venv/bin/activate && python mcp_server/server.py"]
    }
  }
}
```

Restart Claude Desktop. You'll see "El Paso" in the MCP tools list.

### Claude Code

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "el-paso": {
      "command": "bash",
      "args": ["-c", "cd /home/brianba/projects/sandbox/elpaso && source .venv/bin/activate && python mcp_server/server.py"]
    }
  }
}
```

### Cursor

Add the same MCP configuration in Cursor's settings under MCP servers.

## Using El Paso

Once connected, your AI assistant can call `ask_el_paso` automatically when you ask questions about your systems. You can also ask it directly:

### Example Queries

**General questions:**
- "How does the automated shipping system work?"
- "What is the FG allocation process?"
- "How do we handle EDI 850 inbound orders?"

**Code-specific questions (scope: code):**
- "Show me how the ManufacturingOrderService handles order creation"
- "What interfaces does the ShippingService implement?"
- "How is RabbitMQ used in the mes-common library?"

**Documentation-only (scope: docs):**
- "What's the deployment process for PCS Classic?"
- "What are the known risks for the Repricer service?"

**Scoped to a specific repo:**
- "How does authentication work?" (repo: "mes-workflow-ui")

**Scoped to a Confluence space:**
- "What batch jobs run nightly?" (space: "ISS")

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `question` | (required) | Your question |
| `scope` | `"all"` | `"all"`, `"code"` (source code only), `"docs"` (Confluence + GitHub docs/issues/PRs) |
| `repo` | `""` | Filter to a specific GitHub repo (e.g. `"mes-shipping-service"`) |
| `space` | `""` | Filter to a Confluence space key (e.g. `"ISS"`) |

## What's Indexed

| Source | Content | Count |
|---|---|---|
| Confluence (ISS space) | All pages — process docs, service descriptions, troubleshooting | ~1,583 pages |
| GitHub Docs | README.md + /docs folders from mes-* repos | ~140 doc files |
| GitHub Issues | Open and recent issues with comments | Last 12 months |
| GitHub PRs | Merged PR descriptions | Last 12 months |
| GitHub Code | C# source files parsed at class/method boundaries | ~2,856 files |

## How Answers Work

1. Your question is converted to a vector embedding
2. The most relevant chunks are retrieved from the knowledge base
3. Qwen3 (local LLM) synthesizes an answer using only those chunks
4. Every claim is cited with `[Source N]` notation pointing to the original document/file

El Paso will **never hallucinate** — if the answer isn't in the indexed content, it will say so.

## Prerequisites

These must be running on the host machine for El Paso to work:

- **Docker Desktop** with WSL integration enabled (runs Qdrant vector database)
- **Ollama** (runs the embedding model and LLM locally)

Both auto-start if Docker Desktop is set to launch on login and Ollama is installed as a service.

## Keeping Data Fresh

### Incremental Ingestion (recommended)

Run these when content changes in Confluence or GitHub. Only new/modified items get processed — unchanged items are skipped automatically:

```bash
cd ~/projects/sandbox/elpaso
source .venv/bin/activate

python scripts/ingest_confluence.py     # Confluence pages
python scripts/ingest_github_docs.py    # READMEs, issues, PRs
python scripts/ingest_github_code.py    # C# source code
```

A second run immediately after will complete in seconds since nothing has changed.

### Full Rebuild

If data gets messy (duplicates, stale entries), drop the collection and re-ingest everything from scratch:

```bash
python scripts/rebuild_collection.py
```

This takes 60-90 minutes due to GitHub API rate limits.

## Checking Health

```bash
python smoke_test.py                    # Validates Ollama + Qdrant + embeddings
docker compose ps                       # Qdrant container status
ollama list                             # Available models
```

## Viewing Logs

Structured JSON logs are written per day:

```bash
cat logs/elpaso-2026-03-10.jsonl
```

Logs capture ingestion runs (items processed/skipped/failed, duration) and queries (question, scope, result count, latency).
