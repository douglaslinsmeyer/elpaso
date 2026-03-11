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
def ask_el_paso(question: str, scope: str = "all") -> str:
    """Ask El Paso about Ping's software systems, architecture, and processes.

    Searches across Confluence docs, GitHub repos, issues, PRs, and C# source
    code to provide answers with source citations.

    Args:
        question: Your question about the codebase, architecture, or processes
        scope: Filter sources - "all" (default), "code" (source code only), "docs" (documentation only)
    """
    if scope not in ("all", "code", "docs"):
        return f"Invalid scope '{scope}'. Use 'all', 'code', or 'docs'."

    retriever = _get_retriever()
    return retriever.ask(question, scope=scope)


if __name__ == "__main__":
    mcp.run()
