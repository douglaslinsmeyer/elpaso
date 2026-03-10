"""Semantic search test — embed a question, query Qdrant, print results with sources."""

import os
import sys

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embedder import Embedder
from pipeline.store import VectorStore


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python scripts/query_test.py \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    embed_model = config["embedding"]["model"]
    collection_name = config["qdrant"]["collection_name"]
    top_k = config["retrieval"]["top_k"]
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    embedder = Embedder(model=embed_model, ollama_url=ollama_url)
    store = VectorStore(collection_name=collection_name, host=qdrant_host, port=qdrant_port)

    print(f"Question: {question}")
    print(f"Searching top {top_k} results...\n")

    query_vector = embedder.embed(question)
    results = store.search(query_vector, top_k=top_k)

    if not results:
        print("No results found. Is the collection populated?")
        sys.exit(1)

    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        title = result.get("page_title", "unknown")
        heading = result.get("heading_context", "")
        source = result.get("source_type", "unknown")
        url = result.get("page_url", "")
        text = result.get("text", "")

        print(f"--- Result {i} (score: {score:.4f}) ---")
        print(f"Source: [{source}] {title}")
        if heading:
            print(f"Section: {heading}")
        if url:
            print(f"URL: {url}")
        print(f"Text:\n{text[:500]}")
        print()


if __name__ == "__main__":
    main()
