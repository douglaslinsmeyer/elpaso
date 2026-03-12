"""Ingest GitHub source code — fetch, parse with Tree-sitter, embed, store."""

import logging
import os
import sys
import time

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.github_code import GitHubCodeConnector
from pipeline.code_chunker import chunk_code
from pipeline.embedder import Embedder
from pipeline.fingerprint import content_fingerprint
from pipeline.ingestion_tracker import IngestionTracker
from pipeline.logger import get_logger, log_with_data
from pipeline.store import VectorStore

logger = get_logger("el_paso.ingest.github_code")


def run_github_code_ingestion(config: dict, tracker: IngestionTracker, embedder: Embedder, store: VectorStore) -> dict:
    """Run GitHub code ingestion. Returns stats dict with files, chunks, skipped, errors."""
    github_token = os.environ.get("GITHUB_TOKEN")
    github_org = os.environ.get("GITHUB_ORG")
    if not all([github_token, github_org]):
        raise RuntimeError("Set GITHUB_TOKEN and GITHUB_ORG in .env")

    repo_prefix = config.get("github", {}).get("repo_prefix", "")
    extensions = config.get("github", {}).get("code_extensions", [".cs"])
    skip_patterns = config.get("github", {}).get("skip_patterns", [])
    chunk_size = config.get("chunking", {}).get("chunk_size", 512)

    connector = GitHubCodeConnector(
        github_token, github_org, repo_prefix, extensions, skip_patterns
    )

    logger.info("Fetching source files from GitHub...")
    files = connector.fetch_code()
    logger.info(f"Found {len(files)} source files across matching repos")

    total_chunks = 0
    skipped = 0
    errors = 0
    current_ids = set()

    for code_file in files:
        file_id = f"{code_file.repo_name}/{code_file.file_path}"
        current_ids.add(file_id)
        fingerprint = content_fingerprint(code_file.content)

        if not tracker.has_changed("github_code", file_id, fingerprint):
            skipped += 1
            continue

        try:
            store.delete_by_filter(
                source_type="github_code",
                repo_name=code_file.repo_name,
                file_path=code_file.file_path,
            )

            chunks = chunk_code(code_file.content, code_file.language, chunk_size=chunk_size)
            if not chunks:
                tracker.mark_ingested("github_code", file_id, fingerprint)
                continue

            texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed_batch(texts)

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
            tracker.mark_ingested("github_code", file_id, fingerprint)
            logger.info(f"  [{code_file.repo_name}/{code_file.file_path}] → {count} chunks")
        except Exception as e:
            errors += 1
            logger.error(f"  [{code_file.repo_name}/{code_file.file_path}] → ERROR: {e}")

    # Clean up deleted files
    for old_id in tracker.get_all_keys("github_code") - current_ids:
        parts = old_id.split("/", 1)
        if len(parts) == 2:
            store.delete_by_filter(source_type="github_code", repo_name=parts[0], file_path=parts[1])
        tracker.remove("github_code", old_id)

    return {"files": len(files), "chunks": total_chunks, "skipped": skipped, "errors": errors}


def main():
    load_dotenv()
    start = time.time()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    embed_model = config["embedding"]["model"]
    collection_name = config["qdrant"]["collection_name"]
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    embedder = Embedder(model=embed_model, ollama_url=ollama_url)
    store = VectorStore(collection_name=collection_name, host=qdrant_host, port=qdrant_port)
    tracker = IngestionTracker()

    store.ensure_collection(vector_size=embedder.vector_size())

    stats = run_github_code_ingestion(config, tracker, embedder, store)

    tracker.save()
    elapsed = time.time() - start

    log_with_data(
        logger, logging.INFO,
        f"Done: {stats['files']} files → {stats['chunks']} ingested, {stats['skipped']} skipped, {stats['errors']} errors ({elapsed:.1f}s)",
        source="github_code", **stats, elapsed_seconds=round(elapsed, 1),
    )

    try:
        info = store.collection_info()
        logger.info(f"Collection: {info['name']} | Points: {info['points_count']} | Status: {info['status']}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
