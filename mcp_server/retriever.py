"""Retrieval and synthesis pipeline — search Qdrant, synthesize with Qwen3."""

import os

import requests
import yaml

from pipeline.embedder import Embedder
from pipeline.store import VectorStore
from mcp_server.prompts import SYSTEM_PROMPT, build_synthesis_prompt


SCOPE_MAP = {
    "code": ["github_code"],
    "docs": ["confluence", "github_docs", "github_issue", "github_pr"],
    "all": None,
}


class Retriever:
    """Embeds a question, searches Qdrant, synthesizes an answer with Qwen3."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        embed_model = config["embedding"]["model"]
        llm_model = config["llm"]["model"]
        collection_name = config["qdrant"]["collection_name"]
        self.top_k = config["retrieval"]["top_k"]
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

    def ask(self, question: str, scope: str = "all") -> str:
        """Full pipeline: embed → search → synthesize → return answer."""
        source_types = SCOPE_MAP.get(scope, None)

        # Retrieve relevant chunks
        query_vector = self.embedder.embed(question)
        chunks = self.store.search(
            query_vector, top_k=self.top_k, source_types=source_types
        )

        if not chunks:
            return "No relevant information found in the knowledge base."

        # Synthesize answer
        user_prompt = build_synthesis_prompt(question, chunks)
        answer = self._call_llm(SYSTEM_PROMPT, user_prompt)

        return answer
