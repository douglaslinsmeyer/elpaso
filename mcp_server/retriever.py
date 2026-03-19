"""Retrieval and synthesis pipeline — search Qdrant, synthesize with Qwen3."""

import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import requests
import yaml

from pipeline.chunker import Chunk, chunk_text
from pipeline.embedder import Embedder
from pipeline.logger import get_logger, log_with_data
from pipeline.store import VectorStore
from mcp_server.prompts import SYSTEM_PROMPT, build_synthesis_prompt

logger = get_logger("el_paso.retriever")

SCOPE_MAP = {
    "code": ["github_code"],
    "docs": ["confluence", "github_docs", "community"],
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

        chunks = _deduplicate_chunks(raw_chunks)

        # Filter out expired community context (TTL)
        now = datetime.now(timezone.utc).isoformat()
        chunks = [
            c for c in chunks
            if not c.get("expires_at") or c["expires_at"] > now
        ]

        chunks = chunks[:effective_top_k]

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

    def store(
        self,
        text: str,
        title: str,
        author: str = "",
        tags: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> dict:
        """Store community context into the knowledge base."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace-only.")

        identifier = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Chunk long text, wrap short text as a single chunk
        if len(text) > 1500:
            chunks = chunk_text(text)
        else:
            chunks = [Chunk(text=text.strip(), chunk_index=0, heading_context="")]

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_batch(texts)

        expires_at = None
        if expires_in_days is not None:
            expires_at = (now + timedelta(days=expires_in_days)).isoformat()

        payloads = []
        for i, chunk in enumerate(chunks):
            payload = {
                "source_type": "community",
                "text": chunk.text,
                "title": title,
                "identifier": identifier,
                "stored_at": now.isoformat(),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if author:
                payload["author"] = author
            if tags:
                payload["tags"] = tags
            if expires_at:
                payload["expires_at"] = expires_at
            payloads.append(payload)

        self.store.ensure_collection(self.embedder.vector_size())
        count = self.store.upsert_chunks(vectors, payloads)

        log_with_data(
            logger, logging.INFO, "Community context stored",
            identifier=identifier, title=title, chunks_stored=count,
        )

        result = {
            "identifier": identifier,
            "title": title,
            "chunks_stored": count,
        }
        if expires_at:
            result["expires_at"] = expires_at
        return result

    def delete(self, identifier: str) -> dict:
        """Delete community context by identifier."""
        self.store.ensure_collection(self.embedder.vector_size())
        self.store.delete_by_filter(source_type="community", identifier=identifier)

        log_with_data(
            logger, logging.INFO, "Community context deleted",
            identifier=identifier,
        )

        return {"identifier": identifier, "deleted": True}

    def list_stored(self, tag: str = "", limit: int = 50) -> dict:
        """List stored community context entries grouped by identifier."""
        self.store.ensure_collection(self.embedder.vector_size())

        # Scroll enough points to cover multi-chunk entries
        filter_kwargs = {"source_type": "community"}
        payloads = self.store.scroll_by_filter(limit=limit * 10, **filter_kwargs)

        # Group by identifier
        groups: dict[str, dict] = {}
        for p in payloads:
            ident = p.get("identifier", "")
            if not ident:
                continue
            # Tag filter
            if tag and tag not in (p.get("tags") or []):
                continue
            if ident not in groups:
                groups[ident] = {
                    "identifier": ident,
                    "title": p.get("title", ""),
                    "author": p.get("author"),
                    "tags": p.get("tags"),
                    "stored_at": p.get("stored_at"),
                    "expires_at": p.get("expires_at"),
                    "chunk_count": 0,
                }
            groups[ident]["chunk_count"] += 1

        entries = sorted(groups.values(), key=lambda e: e.get("stored_at", ""), reverse=True)[:limit]
        return {"entries": entries, "count": len(entries)}

    def _fuzzy_class_candidates(self, class_name: str, repo: str = "", limit: int = 10) -> list[str]:
        """Find candidate class names via keyword search when exact match fails."""
        keyword_results = self.store.keyword_search(
            class_name,
            top_k=limit,
            source_types=["github_code"],
            repo_name=repo,
        )
        # Extract unique class names from results
        seen: set[str] = set()
        candidates: list[str] = []
        for r in keyword_results:
            cn = r.get("class_name", "")
            if cn and cn != class_name and cn not in seen:
                seen.add(cn)
                candidates.append(cn)
        return candidates

    def get_class(self, class_name: str, repo: str = "") -> dict:
        """Retrieve all chunks for a class or interface, ordered by chunk_index."""
        self.store.ensure_collection(self.embedder.vector_size())

        filter_kwargs: dict = {"source_type": "github_code", "class_name": class_name}
        if repo:
            filter_kwargs["repo_name"] = repo

        payloads = self.store.scroll_by_filter(limit=100, **filter_kwargs)

        if not payloads:
            # Fuzzy fallback: keyword search for similar class names
            candidates = self._fuzzy_class_candidates(class_name, repo=repo)
            if len(candidates) == 1:
                # Auto-resolve single match
                log_with_data(
                    logger, logging.INFO, "Class auto-resolved via fuzzy match",
                    query=class_name, resolved=candidates[0],
                )
                return self.get_class(candidates[0], repo=repo)
            return {
                "class_name": class_name,
                "found": False,
                "candidates": candidates if candidates else [],
            }

        payloads.sort(key=lambda p: p.get("chunk_index", 0))

        first = payloads[0]
        chunks = [
            {
                "method_name": p.get("method_name", ""),
                "chunk_index": p.get("chunk_index", 0),
                "text": p.get("text", ""),
            }
            for p in payloads
        ]

        log_with_data(
            logger, logging.INFO, "Class retrieved",
            class_name=class_name, repo=repo, chunks=len(chunks),
        )

        return {
            "class_name": class_name,
            "found": True,
            "repo_name": first.get("repo_name", ""),
            "file_path": first.get("file_path", ""),
            "namespace": first.get("namespace", ""),
            "is_interface": first.get("is_interface", False),
            "implements_interfaces": first.get("implements_interfaces", []),
            "total_chunks": first.get("total_chunks", len(chunks)),
            "chunks": chunks,
        }

    def find_implementations(self, interface_name: str, repo: str = "") -> dict:
        """Find the interface definition and all classes implementing it."""
        self.store.ensure_collection(self.embedder.vector_size())

        # Normalize: try both with and without "I" prefix
        candidates = [interface_name]
        if not interface_name.startswith("I"):
            candidates.insert(0, f"I{interface_name}")
        elif len(interface_name) > 1:
            candidates.append(interface_name[1:])

        # Step 1: Find the interface definition
        interface_def = None
        resolved_name = interface_name
        for name in candidates:
            filter_kwargs: dict = {"source_type": "github_code", "class_name": name}
            if repo:
                filter_kwargs["repo_name"] = repo
            payloads = self.store.scroll_by_filter(limit=10, **filter_kwargs)
            iface_chunks = [p for p in payloads if p.get("is_interface")]
            if iface_chunks:
                resolved_name = name
                interface_def = {
                    "text": iface_chunks[0].get("text", ""),
                    "repo_name": iface_chunks[0].get("repo_name", ""),
                    "file_path": iface_chunks[0].get("file_path", ""),
                    "namespace": iface_chunks[0].get("namespace", ""),
                }
                break

        # Step 2: Find implementations
        impl_filter: dict = {"source_type": "github_code", "implements_interfaces": resolved_name}
        if repo:
            impl_filter["repo_name"] = repo
        impl_payloads = self.store.scroll_by_filter(limit=200, **impl_filter)

        # Group by (class_name, repo_name, file_path)
        groups: dict[tuple, dict] = {}
        for p in impl_payloads:
            key = (p.get("class_name", ""), p.get("repo_name", ""), p.get("file_path", ""))
            if key not in groups:
                groups[key] = {
                    "class_name": key[0],
                    "repo_name": key[1],
                    "file_path": key[2],
                    "namespace": p.get("namespace", ""),
                    "methods": [],
                }
            method = p.get("method_name", "")
            if method and method not in groups[key]["methods"]:
                groups[key]["methods"].append(method)

        implementations = sorted(groups.values(), key=lambda g: (g["repo_name"], g["class_name"]))

        log_with_data(
            logger, logging.INFO, "Implementations found",
            interface_name=resolved_name, count=len(implementations),
        )

        return {
            "interface_name": resolved_name,
            "interface": interface_def,
            "implementations": implementations,
            "count": len(implementations),
        }

    def discover_repos(self, query: str, scope: str = "all", top_k: int = 30) -> dict:
        """Search broadly and aggregate results by repository."""
        chunks = self.search(query, scope=scope, top_k=top_k, mode="hybrid")

        # Aggregate code results by repo
        repo_data: dict[str, dict] = {}
        docs: list[dict] = []

        for c in chunks:
            source_type = c.get("source_type", "")
            repo_name = c.get("repo_name", "")

            if repo_name:
                if repo_name not in repo_data:
                    repo_data[repo_name] = {
                        "repo_name": repo_name,
                        "hits": 0,
                        "sample_files": [],
                        "sample_classes": [],
                    }
                rd = repo_data[repo_name]
                rd["hits"] += 1
                fp = c.get("file_path", "")
                if fp and fp not in rd["sample_files"] and len(rd["sample_files"]) < 5:
                    rd["sample_files"].append(fp)
                cn = c.get("class_name", "")
                if cn and cn not in rd["sample_classes"] and len(rd["sample_classes"]) < 5:
                    rd["sample_classes"].append(cn)
            elif source_type in ("confluence", "github_docs", "community"):
                docs.append({
                    "title": c.get("page_title") or c.get("title", ""),
                    "url": c.get("page_url", ""),
                    "source_type": source_type,
                })

        repos = sorted(repo_data.values(), key=lambda r: r["hits"], reverse=True)

        log_with_data(
            logger, logging.INFO, "Repos discovered",
            query=query, repo_count=len(repos),
        )

        return {"query": query, "repos": repos, "docs": docs}
