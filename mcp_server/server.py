"""El Paso MCP Server — exposes search tools via Model Context Protocol."""

import os
import sys

import yaml
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Load server config
_config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)
with open(_config_path) as _f:
    _server_cfg = yaml.safe_load(_f).get("server", {})

_transport = os.environ.get("MCP_TRANSPORT", _server_cfg.get("transport", "stdio"))
_host = _server_cfg.get("host", "127.0.0.1")
_port = int(_server_cfg.get("port", 8080))

_instructions = """\
El Paso is a RAG-powered knowledge base for Ping Golf's manufacturing software systems.

## What's indexed
- **Confluence docs** from the ISS space (processes, architecture, how-tos, infrastructure decisions)
- **C# source code** (`.cs` files only) from mes-* GitHub repos, parsed at class/method granularity \
via Tree-sitter. Each chunk is a single method or small class with namespace/class/method context headers
- **GitHub markdown docs** (READMEs, docs/ folders) from mes-* repos
- **Community context** — tribal knowledge stored by users via `store_context`

## What's NOT indexed
- GitHub issues and PRs
- Non-C# files: YAML (K8s manifests, Helm charts), Terraform, SQL migrations, Dockerfiles. \
For infrastructure and deployment questions, use `search_docs` instead of `search_code`
- Only repos matching the `mes-*` prefix are indexed

## Indexed repos (not just microservices)
- Service APIs: mes-api-rest, mes-api-workflow, mes-api-manufacturing-order-loader, etc.
- Frontend apps: mes-workflow, mes-wgl, mes-wgl-app
- Client libraries: mes-api-rest-client, mes-api-workflow-client, mes-api-manufacturing-order-service-client
- Infrastructure/support: mes-database-janitor, mes-manufacturing-order-process-recorder

## Tool selection
- Know the exact class/method/interface name? → `search_code` with mode="keyword" (most reliable)
- Found a class name and want its full definition? → `get_class` (returns all methods in order, \
auto-resolves fuzzy matches)
- Want to trace interface → implementations? → `find_implementations`
- Don't know which repo to look in? → `discover_repos`
- Looking for code by concept? → `search_code` with mode="semantic" (lower precision)
- Looking for documentation, processes, architecture, infrastructure? → `search_docs`
- Need to search across both code and docs? → `search_el_paso`
- Store/manage tribal knowledge? → `store_context` / `list_context` / `delete_context`

## Search strategy tips
- For code: use single exact identifiers with keyword mode. Multi-word keyword queries often return \
0 results. Avoid hybrid mode — it introduces semantic noise that degrades code results.
- For docs: natural language questions work well with semantic mode (the default).
- Investigate iteratively: Controller → Service interface → Repository → Contract → docs for \
architecture context. One search rarely gives the full picture.
- For infrastructure questions (K8s, database topology, Terraform): use `search_docs` since \
infrastructure-as-code files aren't in the code index.
- Check the `source_type` field in results to identify origin: github_code, confluence, \
github_docs, community.\
"""

if _transport == "streamable-http":
    # Build allowed origins from env var (comma-separated) or default to permissive
    _allowed_origins_str = os.environ.get("MCP_ALLOWED_ORIGINS", "")
    if _allowed_origins_str:
        _allowed_origins = [o.strip() for o in _allowed_origins_str.split(",")]
    else:
        _allowed_origins = ["*"]

    _security = TransportSecuritySettings(
        enable_dns_rebinding_protection=_allowed_origins != ["*"],
        allowed_origins=_allowed_origins,
    )

    mcp = FastMCP(
        "El Paso",
        instructions=_instructions,
        host=_host,
        port=_port,
        json_response=True,
        transport_security=_security,
    )
else:
    mcp = FastMCP("El Paso", instructions=_instructions)

# Lazy-init retriever (only when first tool call happens)
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from mcp_server.retriever import Retriever
        _retriever = Retriever()
    return _retriever


@mcp.tool()
def search_code(
    query: str,
    repo: str = "",
    top_k: int = 8,
    mode: str = "",
) -> dict:
    """Search C# source code from Ping's mes-* GitHub repositories.

    Searches `.cs` files only, parsed at class/method granularity via Tree-sitter. Each result
    is a single method or small class, prefixed with context: `// Namespace: X / // Class: Y : IY / // Method: Z`.

    Indexed repos include service APIs (mes-api-rest, mes-api-workflow, mes-api-manufacturing-order-loader),
    frontend Blazor apps (mes-workflow, mes-wgl), client libraries (mes-api-rest-client,
    mes-api-workflow-client), and infrastructure services (mes-database-janitor).

    NOT indexed: YAML, Terraform, SQL, Dockerfiles — use search_docs for infrastructure topics.

    Query guidance (based on empirical testing):
    - Exact identifiers (class/method/interface name): use mode="keyword" — most reliable
    - Conceptual queries ("retry pattern", "how does X work"): use mode="semantic" — less precise
    - mode="hybrid" (default) can introduce irrelevant semantic matches — prefer explicit keyword or semantic
    - Multi-word keyword queries often return 0 — use single identifiers
    - Investigate iteratively: find a Controller → search its Service interface → Repository → Contract

    Result fields: text, score, repo_name, file_path, class_name, method_name, namespace,
    is_interface, implements_interfaces, chunk_index, total_chunks, rrf_score (hybrid only).
    Keyword mode returns score=0.0 (normal — ranked by Qdrant BM25, not vector similarity).

    Args:
        query: Class name, method name, interface name, or conceptual description
        repo: Filter to a specific repo (e.g. "mes-api-workflow", "mes-database-janitor", \
"mes-api-rest-client"). All repos use the "mes-" prefix.
        top_k: Number of results (default 8, beyond ~12 rarely helps)
        mode: "keyword" (exact identifier match — best for code), "semantic" (conceptual), \
or "hybrid" (default — can be noisy). Prefer explicit keyword or semantic.
    """
    retriever = _get_retriever()
    chunks = retriever.search(query, scope="code", repo=repo, top_k=top_k, mode=mode or None)
    return {"query": query, "result_count": len(chunks), "results": chunks}


@mcp.tool()
def search_docs(
    query: str,
    space: str = "",
    top_k: int = 8,
    mode: str = "",
) -> dict:
    """Search Confluence documentation, GitHub markdown docs, and community-stored context.

    Covers the ISS Confluence space (processes, architecture, how-tos, infrastructure decisions),
    README/docs from mes-* repos, and community-stored tribal knowledge.

    Also useful for infrastructure topics: K8s manifests, Terraform config, database architecture,
    and deployment patterns are often documented in Confluence even though the raw infra-as-code
    files aren't in the code index.

    Semantic mode (default) works well for natural language questions about processes and architecture.
    Keyword mode works well for specific terms (e.g. "workgen", "shiplinx", "MWS410MI").

    Result fields vary by source_type:
    - confluence: text, score, page_title, page_url, space_key, heading_context, author, last_modified
    - github_docs: text, score, repo_name, file_path
    - community: text, score, title, tags, identifier, author, stored_at, expires_at

    Args:
        query: Natural language question, or specific term to search for
        space: Filter to a Confluence space key. Currently the only indexed space is "ISS".
        top_k: Number of results (default 8, beyond ~12 rarely helps)
        mode: "semantic" (default — best for natural language), "keyword" (specific terms), or "hybrid"
    """
    retriever = _get_retriever()
    chunks = retriever.search(query, scope="docs", space=space, top_k=top_k, mode=mode or None)
    return {"query": query, "result_count": len(chunks), "results": chunks}


@mcp.tool()
def search_el_paso(
    query: str,
    scope: str = "all",
    repo: str = "",
    space: str = "",
    top_k: int = 8,
    mode: str = "",
) -> dict:
    """Search across all of El Paso's knowledge base — code, docs, and community context.

    Use this when you don't know which scope contains the answer, or need results from
    multiple source types at once. The scoped tools (search_code, search_docs) are preferred
    when you know the scope — they have better defaults and more targeted guidance.

    Scope-to-source mapping:
    - "all" (default): searches everything
    - "code": C# source code (github_code)
    - "docs": Confluence + GitHub markdown + community context
    - "confluence": Confluence pages only

    When scope="all", results may mix source types. Check the source_type field to distinguish.

    Args:
        query: Search query — identifier, natural language question, or specific term
        scope: "all" (default), "code", "docs", or "confluence"
        repo: Filter to a specific GitHub repo (e.g. "mes-api-workflow"). All repos use the "mes-" prefix.
        space: Filter to a Confluence space key. Currently the only indexed space is "ISS".
        top_k: Number of results (default 8, beyond ~12 rarely helps)
        mode: "keyword" (exact match), "semantic" (conceptual), or "hybrid". Default depends on scope.
    """
    retriever = _get_retriever()
    chunks = retriever.search(query, scope=scope, repo=repo, space=space, top_k=top_k, mode=mode or None)
    return {
        "query": query,
        "result_count": len(chunks),
        "results": chunks,
    }


@mcp.tool()
def get_class(
    class_name: str,
    repo: str = "",
) -> dict:
    """Get the complete definition of a C# class or interface — all methods in order.

    Use after search_code finds a relevant class name to see its full definition without
    multiple searches. Returns all method chunks sorted by position, plus metadata:
    namespace, implements_interfaces, repo_name, file_path.

    For large classes (many methods), all chunks are returned — check the method_name
    fields to find specific methods of interest.

    If the exact class name isn't found, a fuzzy search runs automatically:
    - Single match: auto-resolves and returns the full class definition
    - Multiple matches: returns found=false with a "candidates" list of similar class names
    - No matches: returns found=false with an empty candidates list

    Args:
        class_name: Class or interface name (e.g. "DeliveryController", "IDeliveryService"). \
Exact matches are preferred, but close matches will be suggested if no exact match is found.
        repo: Filter to a specific repo (optional, useful when same class name exists in multiple repos). \
All repos use the "mes-" prefix.
    """
    retriever = _get_retriever()
    return retriever.get_class(class_name, repo=repo)


@mcp.tool()
def find_implementations(
    interface_name: str,
    repo: str = "",
) -> dict:
    """Find all C# classes that implement a given interface.

    Returns the interface definition (if indexed) and all implementing classes with their
    repo, file path, namespace, and method names. Useful for tracing SOLID patterns:
    IDeliveryService → DeliveryService, MockDeliveryService, etc.

    Only direct implementations are tracked (from Tree-sitter parsing of C# base type lists).

    Args:
        interface_name: Interface name (e.g. "IDeliveryService"). "I" prefix is conventional \
but not required — both "IFoo" and "Foo" are searched.
        repo: Filter to a specific repo (optional). Omit to search all indexed repos.
    """
    retriever = _get_retriever()
    return retriever.find_implementations(interface_name, repo=repo)


@mcp.tool()
def discover_repos(
    query: str,
    scope: str = "all",
) -> dict:
    """Discover which repositories are relevant to a domain concept or keyword.

    Runs a broad search and aggregates results by repo, showing hit count and sample
    file paths / class names per repo. Use when you don't know which repo to look in.

    Also returns relevant documentation (Confluence pages, GitHub docs) as a secondary section.

    Examples: "ManufacturingOrder", "shipping label", "dead letter queue"

    Args:
        query: Domain concept, keyword, or identifier to search for
        scope: "code" (repos only), "docs" (documentation), or "all" (default)
    """
    retriever = _get_retriever()
    return retriever.discover_repos(query, scope=scope)


@mcp.tool()
def store_context(
    text: str,
    title: str,
    author: str = "",
    tags: list[str] | None = None,
    expires_in_days: int | None = None,
) -> dict:
    """Store community context (tribal knowledge, decisions, notes) into El Paso's knowledge base.

    Stored content becomes searchable via search_docs (scope "docs") and search_el_paso alongside
    formal documentation. Multiple people storing similar context creates a preponderance signal.

    Args:
        text: The context to store (any length — will be chunked if needed)
        title: A descriptive label for this context (e.g. "Shipping service retry pattern")
        author: Who is storing this (optional, helps with provenance)
        tags: Categorization tags (optional, e.g. ["shipping", "retry"])
        expires_in_days: Auto-expire after N days (recommended for time-sensitive context)
    """
    if not text or not text.strip():
        return {"error": "Text cannot be empty."}
    if not title or not title.strip():
        return {"error": "Title is required."}

    retriever = _get_retriever()
    return retriever.store(
        text=text, title=title, author=author, tags=tags,
        expires_in_days=expires_in_days,
    )


@mcp.tool()
def delete_context(identifier: str) -> dict:
    """Delete previously stored community context by its identifier.

    Args:
        identifier: The UUID identifier returned by store_context when the content was stored
    """
    retriever = _get_retriever()
    return retriever.delete(identifier)


@mcp.tool()
def list_context(
    tag: str = "",
    limit: int = 50,
) -> dict:
    """List stored community context entries, sorted newest-first.

    Returns entries grouped by identifier with metadata. The chunk_count field shows
    how many vector chunks each entry was split into during storage.

    Args:
        tag: Filter to entries with this tag (optional)
        limit: Maximum entries to return (default 50)
    """
    retriever = _get_retriever()
    return retriever.list_stored(tag=tag, limit=limit)


if __name__ == "__main__":
    mcp.run(transport=_transport)
