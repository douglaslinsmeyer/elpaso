"""El Paso MCP Server — exposes ask_el_paso tool via Model Context Protocol."""

import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

mcp = FastMCP(
    "El Paso",
    instructions=(
        "El Paso is a RAG-powered knowledge base for Ping Golf's manufacturing "
        "software systems. It has indexed Confluence documentation, GitHub docs, "
        "issues, PRs, and C# source code from the mes-* microservices."
    ),
)

# Lazy-init retriever (only when first tool call happens)
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from mcp_server.retriever import Retriever
        _retriever = Retriever()
    return _retriever


@mcp.tool()
def ask_el_paso(
    question: str,
    scope: str = "all",
    repo: str = "",
    space: str = "",
) -> str:
    """Ask El Paso about Ping's software systems, architecture, and processes.

    Searches across Confluence docs, GitHub repos, issues, PRs, and C# source
    code to provide answers with source citations.

    Args:
        question: Your question about the codebase, architecture, or processes
        scope: Filter sources - "all" (default), "code" (source code only), "docs" (documentation only), "issues" (GitHub issues/PRs), "confluence" (Confluence only)
        repo: Filter to a specific GitHub repo name (e.g. "mes-shipping-service")
        space: Filter to a specific Confluence space key (e.g. "ISS")
    """
    valid_scopes = ("all", "code", "docs", "issues", "confluence")
    if scope not in valid_scopes:
        return f"Invalid scope '{scope}'. Use one of: {', '.join(valid_scopes)}."

    retriever = _get_retriever()
    return retriever.ask(question, scope=scope, repo=repo, space=space)


@mcp.tool()
def search_code(
    query: str,
    repo: str = "",
    top_k: int = 8,
    mode: str = "",
) -> dict:
    """Search C# source code indexed from GitHub repositories.

    Returns raw code chunks with class/method/namespace metadata and relevance scores.
    Defaults to hybrid search (semantic + keyword) for better identifier matching.

    Args:
        query: Search query (e.g. "ProcessShipmentLabel", "RabbitMQ consumer pattern")
        repo: Filter to a specific repo name (e.g. "mes-shipping-service")
        top_k: Number of results to return (default 8)
        mode: Search mode - "semantic", "keyword", or "hybrid" (default: "hybrid" for code)
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
    """Search Confluence documentation and GitHub markdown docs.

    Returns documentation chunks with page titles, URLs, and relevance scores.

    Args:
        query: Search query about processes, architecture, or documentation
        space: Filter to a specific Confluence space key (e.g. "ISS")
        top_k: Number of results to return (default 8)
        mode: Search mode - "semantic", "keyword", or "hybrid" (default: "semantic")
    """
    retriever = _get_retriever()
    chunks = retriever.search(query, scope="docs", space=space, top_k=top_k, mode=mode or None)
    return {"query": query, "result_count": len(chunks), "results": chunks}


@mcp.tool()
def search_issues(
    query: str,
    repo: str = "",
    top_k: int = 8,
    mode: str = "",
) -> dict:
    """Search GitHub issues and merged pull requests.

    Returns issue/PR chunks with titles, authors, and relevance scores.

    Args:
        query: Search query about bugs, features, or PR discussions
        repo: Filter to a specific repo name (e.g. "mes-shipping-service")
        top_k: Number of results to return (default 8)
        mode: Search mode - "semantic", "keyword", or "hybrid" (default: "semantic")
    """
    retriever = _get_retriever()
    chunks = retriever.search(query, scope="issues", repo=repo, top_k=top_k, mode=mode or None)
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
    """Search El Paso's knowledge base and return raw ranked chunks with scores.

    Returns chunks with full metadata for the consuming agent to synthesize.
    Use this instead of ask_el_paso when you want raw retrieval results.

    Args:
        query: Search query about the codebase, architecture, or processes
        scope: Filter sources - "all" (default), "code", "docs", "issues", "confluence"
        repo: Filter to a specific GitHub repo name (e.g. "mes-shipping-service")
        space: Filter to a specific Confluence space key (e.g. "ISS")
        top_k: Number of results to return (default 8)
        mode: Search mode - "semantic", "keyword", or "hybrid" (default depends on scope)
    """
    retriever = _get_retriever()
    chunks = retriever.search(query, scope=scope, repo=repo, space=space, top_k=top_k, mode=mode or None)
    return {
        "query": query,
        "result_count": len(chunks),
        "results": chunks,
    }


if __name__ == "__main__":
    mcp.run()
