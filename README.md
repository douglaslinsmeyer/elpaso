# El Paso

A retrieval-as-a-service system that ingests organizational knowledge from Confluence and GitHub, then exposes it via [MCP](https://modelcontextprotocol.io/) tools for any AI agent to search with source attribution.

Built for a manufacturing-focused microservices environment (C#/.NET, RabbitMQ, PostgreSQL, Blazor). The retrieval system itself is Python. Consuming agents (Claude, Cursor, etc.) bring their own LLM — El Paso handles the search.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Ingestion Layer│────▶│ Processing Layer  │────▶│   Retrieval Layer   │────▶│   MCP Server     │
│  (connectors/)  │     │  (pipeline/)      │     │                     │     │  (mcp_server/)   │
│                 │     │                   │     │                     │     │                  │
│ Confluence      │     │ Chunking          │     │ Semantic search     │     │ search_code()    │
│ GitHub docs     │     │ Embedding (Ollama)│     │ Hybrid search (RRF) │     │ search_docs()    │
│ GitHub code     │     │ Qdrant storage    │     │ Keyword search      │     │ search_issues()  │
│                 │     │ Fingerprinting    │     │ Dedup & ranking     │     │ search_el_paso() │
└─────────────────┘     └──────────────────┘     └─────────────────────┘     └──────────────────┘
```

## Tech Stack

- **Python 3.10+**
- **Ollama** with nomic-embed-text for embeddings
- **Qdrant** (Docker) for vector storage
- **LlamaIndex** for text chunking
- **Tree-sitter** for C# code parsing
- **Python MCP SDK** for the MCP server

---

## Setup for Knuckleheads (Eric, this means you)

Read every step. Do them in order. Do not skip steps. Do not freestyle.

### Step 1: Install the prerequisites

You need these things installed on your machine BEFORE you touch this repo:

| Thing | Why you need it | How to get it |
|-------|----------------|---------------|
| **Python 3.10+** | Runs everything | `sudo apt install python3 python3-venv python3-pip` (Ubuntu/WSL) |
| **Docker** | Runs the Qdrant vector database | [Install Docker](https://docs.docker.com/get-docker/) — on WSL2, install Docker Desktop on Windows |
| **Ollama** | Runs the embedding model locally | `curl -fsSL https://ollama.ai/install.sh \| sh` |
| **Git** | Clone the repo | You probably have this already |

**How to check you have them:**
```bash
python3 --version    # Should say 3.10 or higher
docker --version     # Should say Docker version 2x.x.x
ollama --version     # Should say ollama version 0.x.x
```

If any of those commands fail, stop and install the missing thing. Do not proceed.

### Step 2: Clone the repo

```bash
git clone <repo-url>
cd elpaso
```

### Step 3: Start Docker

Docker needs to be running. If you're on WSL2, open Docker Desktop on Windows and wait for it to say "Running."

**How to check:**
```bash
docker ps
```

If you get "Cannot connect to the Docker daemon" — Docker is not running. Start it. Wait. Try again.

### Step 4: Start Qdrant

```bash
docker compose up -d
```

**How to check it worked:**
```bash
curl http://localhost:6333/healthz
```

Should print `ok` or similar. If it says "connection refused," Qdrant didn't start. Run `docker compose logs` to see what went wrong.

### Step 5: Pull the embedding model with Ollama

```bash
ollama pull nomic-embed-text    # Embedding model (~275MB)
```

**How to check it's there:**
```bash
ollama list
```

You should see `nomic-embed-text` in the list.

### Step 6: Create the Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**IMPORTANT:** Every time you open a new terminal to work on this project, you need to run:
```bash
source .venv/bin/activate
```

If your terminal prompt doesn't start with `(.venv)`, you forgot. Go back and do it.

### Step 7: Set up your environment variables

```bash
cp .env.example .env
```

Now open `.env` in a text editor and fill in your actual credentials:

```
# Confluence — get the API token from https://id.atlassian.com/manage-profile/security/api-tokens
CONFLUENCE_URL=https://pingis.atlassian.net/wiki
CONFLUENCE_USERNAME=your-actual-email@ping.com
CONFLUENCE_API_TOKEN=your-actual-token

# GitHub — create a token at https://github.com/settings/tokens (needs repo scope)
GITHUB_TOKEN=ghp_your-actual-token-here
GITHUB_ORG=pinggolf

# These are fine as-is for local dev
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_BASE_URL=http://localhost:11434
```

**Do not commit your `.env` file.** It's in `.gitignore` for a reason.

### Step 8: Run the smoke test

This checks that Ollama, Qdrant, and embeddings all work together:

```bash
python smoke_test.py
```

**You need to see "all steps passed."** If any step fails:
- `Ollama health check FAIL` → Ollama isn't running. Run `ollama serve` in another terminal.
- `Qdrant health check FAIL` → Docker/Qdrant isn't running. Go back to Step 4.
- `Embedding test FAIL` → You didn't pull the models. Go back to Step 5.

### Step 9: Ingest the data

This fetches everything from Confluence and GitHub, chunks it, embeds it, and stores it in Qdrant. It takes about 30–40 minutes the first time.

```bash
python scripts/ingest_all.py
```

Or if you want to run just one source:
```bash
python scripts/ingest_all.py --source confluence
python scripts/ingest_all.py --source github_docs
python scripts/ingest_all.py --source github_code
```

**If you see "Collection `el_paso` doesn't exist" errors** — this is the old version. Pull the latest code. The new version creates the collection automatically.

**If GitHub code ingestion fails with 502 errors** — GitHub's API is being flaky. Just run it again:
```bash
python scripts/ingest_all.py --source github_code
```

It picks up where it left off. Already-ingested files get skipped.

### Step 10: Connect to Claude Code

Add this to your Claude Code project (it may already be in `.mcp.json`):

```json
{
  "mcpServers": {
    "el-paso": {
      "command": "/absolute/path/to/elpaso/.venv/bin/python",
      "args": ["mcp_server/server.py"],
      "cwd": "/absolute/path/to/elpaso"
    }
  }
}
```

Replace `/absolute/path/to/elpaso` with the actual path on your machine.

Then in Claude Code, run `/mcp` and verify `el-paso` is connected.

### You're done

Ask Claude anything about Ping's MES systems. It will use El Paso automatically.

---

## Setup for Non-Knuckleheads

```bash
docker compose up -d
ollama pull nomic-embed-text
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env   # fill in credentials
python smoke_test.py
python scripts/ingest_all.py
```

---

## MCP Tools

| Tool | What it does |
|------|-------------|
| `search_el_paso(query, scope?, repo?, space?, top_k?, mode?)` | Raw ranked chunks with scores and metadata |
| `search_code(query, repo?, top_k?, mode?)` | Search C# source code (hybrid search by default) |
| `search_docs(query, space?, top_k?, mode?)` | Search Confluence + GitHub markdown docs |
| `search_issues(query, repo?, top_k?, mode?)` | Search GitHub issues and merged PRs |

**Scopes:** `all`, `code`, `docs`, `issues`, `confluence`

**Search modes:** `semantic` (vector similarity), `keyword` (text/identifier match), `hybrid` (both combined via RRF — default for code)

## Configuration

- `config.yaml` — model names, chunking params, retrieval settings, search mode defaults
- `.env` — API credentials (never commit this)

## Testing

```bash
pytest                        # All tests
pytest tests/test_foo.py      # Single file
pytest -k "test_name"         # By name
```

## Roadmap

See [`docs/ElPaso_Roadmap.md`](docs/ElPaso_Roadmap.md).

## License

Private / Internal Use
