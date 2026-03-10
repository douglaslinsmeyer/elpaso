"""Ingest GitHub source code — fetch, parse with Tree-sitter, embed, store."""

import os
import sys

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.github_code import GitHubCodeConnector
from pipeline.code_chunker import chunk_code
from pipeline.embedder import Embedder
from pipeline.store import VectorStore


def main():
    load_dotenv()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    github_token = os.environ.get("GITHUB_TOKEN")
    github_org = os.environ.get("GITHUB_ORG")
    if not all([github_token, github_org]):
        print("ERROR: Set GITHUB_TOKEN and GITHUB_ORG in .env")
        sys.exit(1)

    repo_prefix = config.get("github", {}).get("repo_prefix", "")
    extensions = config.get("github", {}).get("code_extensions", [".cs"])
    skip_patterns = config.get("github", {}).get("skip_patterns", [])
    chunk_size = config.get("chunking", {}).get("chunk_size", 512)
    embed_model = config["embedding"]["model"]
    collection_name = config["qdrant"]["collection_name"]
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    connector = GitHubCodeConnector(
        github_token, github_org, repo_prefix, extensions, skip_patterns
    )
    embedder = Embedder(model=embed_model, ollama_url=ollama_url)
    store = VectorStore(collection_name=collection_name, host=qdrant_host, port=qdrant_port)

    print("Fetching source files from GitHub...")
    files = connector.fetch_code()
    print(f"Found {len(files)} source files across matching repos")

    total_chunks = 0
    errors = 0
    collection_ready = False

    for code_file in files:
        try:
            chunks = chunk_code(code_file.content, code_file.language, chunk_size=chunk_size)
            if not chunks:
                continue

            texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed_batch(texts)

            if not collection_ready:
                store.ensure_collection(vector_size=len(vectors[0]))
                collection_ready = True

            payloads = [
                {
                    "source_type": "github_code",
                    "repo_name": code_file.repo_name,
                    "repo_url": code_file.repo_url,
                    "file_path": code_file.file_path,
                    "language": code_file.language,
                    "namespace": chunk.namespace,
                    "class_name": chunk.class_name,
                    "method_name": chunk.method_name,
                    "is_interface": chunk.is_interface,
                    "implements_interfaces": chunk.implements_interfaces,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": len(chunks),
                    "text": chunk.text,
                }
                for chunk, _ in zip(chunks, vectors)
            ]

            count = store.upsert_chunks(vectors, payloads)
            total_chunks += count
            print(f"  [{code_file.repo_name}/{code_file.file_path}] → {count} chunks")
        except Exception as e:
            errors += 1
            print(f"  [{code_file.repo_name}/{code_file.file_path}] → ERROR: {e}")

    print(f"\nDone: {len(files)} files → {total_chunks} code chunks stored")
    if errors:
        print(f"  ({errors} files failed — see errors above)")

    info = store.collection_info()
    print(f"Collection: {info['name']} | Points: {info['points_count']} | Status: {info['status']}")


if __name__ == "__main__":
    main()
