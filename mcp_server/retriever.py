"""Retrieval and synthesis pipeline — search Qdrant, synthesize with Qwen3."""

import logging
import os
import time

import requests
import yaml

from pipeline.embedder import Embedder
from pipeline.logger import get_logger, log_with_data
from pipeline.store import VectorStore
from mcp_server.prompts import SYSTEM_PROMPT, build_synthesis_prompt

logger = get_logger("el_paso.retriever")

SCOPE_MAP = {
    "code": ["github_code"],
    "docs": ["confluence", "github_docs"],
    "issues": ["github_issue", "github_pr"],
    "confluence": ["confluence"],
    "all": None,
}


def _deduplicate_chunks(chunks: list[dict], similarity_threshold: float = 0.95) -> list[dict]:
    """Remove near-duplicate chunks based on text overlap."""
    if not chunks:
        return chunks

    seen_texts: list[str] = []
    unique: list[dict] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        is_dup = False
        for seen in seen_texts:
            # Simple overlap check: if >80% of shorter text is contained in longer
            shorter, longer = (text, seen) if len(text) <= len(seen) else (seen, text)
            if shorter and shorter in longer:
                is_dup = True
                break
        if not is_dup:
            seen_texts.append(text)
            unique.append(chunk)

    return unique


class Retriever:
    """Embeds a question, searches Qdrant, synthesizes an answer with Qwen3."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        embed_model = self.config["embedding"]["model"]
        llm_model = self.config["llm"]["model"]
        collection_name = self.config["qdrant"]["collection_name"]
        self.top_k = self.config["retrieval"]["top_k"]
        self.llm_model = llm_model

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

        self.embedder = Embedder(model=embed_model, ollama_url=ollama_url)
        self.store = VectorStore(
            collection_name=collection_name, host=qdrant_host, port=qdrant_port
        )
        self.ollama_url = ollama_url

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call Qwen3 via Ollama chat API."""
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def _resolve_mode(self, mode: str | None, scope: str) -> str:
        """Determine search mode based on explicit mode, scope, and config defaults."""
        if mode:
            return mode
        search_config = self.config.get("search", {})
        if scope == "code":
            return search_config.get("code_default_mode", "hybrid")
        return search_config.get("default_mode", "semantic")

    def search(
        self,
        question: str,
        scope: str = "all",
        repo: str = "",
        space: str = "",
        top_k: int | None = None,
        mode: str | None = None,
    ) -> list[dict]:
        """Embed question, search Qdrant, dedup, return ranked chunks with scores and metadata.

        Args:
            mode: "semantic" (vector only), "keyword" (text match), or "hybrid" (RRF merge).
                  Default: "hybrid" for code scope, "semantic" for others (configurable in config.yaml).
        """
        effective_top_k = top_k or self.top_k
        effective_mode = self._resolve_mode(mode, scope)
        source_types = SCOPE_MAP.get(scope, None)
        rrf_k = self.config.get("search", {}).get("rrf_k", 60)

        query_vector = self.embedder.embed(question)

        if effective_mode == "keyword":
            raw_chunks = self.store.keyword_search(
                question,
                top_k=effective_top_k + 4,
                source_types=source_types,
                repo_name=repo,
                space_key=space,
            )
        elif effective_mode == "hybrid":
            raw_chunks = self.store.hybrid_search(
                query_vector, question,
                top_k=effective_top_k + 4,
                rrf_k=rrf_k,
                source_types=source_types,
                repo_name=repo,
                space_key=space,
            )
        else:  # semantic (default)
            raw_chunks = self.store.search(
                query_vector,
                top_k=effective_top_k + 4,
                source_types=source_types,
                repo_name=repo,
                space_key=space,
            )

        chunks = _deduplicate_chunks(raw_chunks)[:effective_top_k]

        log_with_data(
            logger, logging.INFO, "Search executed",
            question=question, scope=scope, repo=repo, space=space,
            mode=effective_mode, raw_results=len(raw_chunks), deduped_results=len(chunks),
        )

        return chunks

    def ask(self, question: str, scope: str = "all", repo: str = "", space: str = "") -> str:
        """Full pipeline: embed → search → deduplicate → synthesize → return answer."""
        start = time.time()

        chunks = self.search(question, scope=scope, repo=repo, space=space)

        if not chunks:
            return "No relevant information found in the knowledge base."

        user_prompt = build_synthesis_prompt(question, chunks)
        answer = self._call_llm(SYSTEM_PROMPT, user_prompt)

        elapsed = time.time() - start
        log_with_data(
            logger, logging.INFO, "Answer synthesized",
            question=question, elapsed_seconds=round(elapsed, 2),
        )

        return answer
