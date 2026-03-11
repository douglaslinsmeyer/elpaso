"""Embedding wrapper around Ollama's embedding API."""

import requests


class Embedder:
    """Generates embeddings via Ollama's /api/embed endpoint."""

    # nomic-embed-text context window is 8192 tokens (~32k chars)
    MAX_CHARS = 30000

    def __init__(self, model: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")

    def _truncate(self, text: str) -> str:
        if len(text) > self.MAX_CHARS:
            return text[: self.MAX_CHARS]
        return text

    def embed(self, text: str) -> list[float]:
        """Embed a single text string, returning a float vector."""
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": self._truncate(text)},
            timeout=30,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if not embeddings:
            raise ValueError(f"No embeddings returned for text: {text[:50]}...")
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Falls back to one-at-a-time on batch failure."""
        if not texts:
            return []

        # Try batch first
        try:
            truncated = [self._truncate(t) for t in texts]
            resp = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model, "input": truncated},
                timeout=120,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings", [])
            if len(embeddings) == len(texts):
                return embeddings
        except requests.HTTPError:
            pass

        # Fallback: embed one at a time
        return [self.embed(text) for text in texts]

    def vector_size(self) -> int:
        """Return the embedding dimension by probing the model with a short string."""
        probe = self.embed("dimension probe")
        return len(probe)
