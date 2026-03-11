"""Ingest GitHub docs, issues, and PRs — fetch, chunk, embed, store in Qdrant."""

import logging
import os
import sys
import time

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.github_docs import GitHubDocsConnector
from connectors.github_issues import GitHubIssuesConnector
from pipeline.chunker import chunk_text
from pipeline.embedder import Embedder
from pipeline.ingestion_tracker import IngestionTracker
from pipeline.logger import get_logger, log_with_data
from pipeline.store import VectorStore

logger = get_logger("el_paso.ingest.github_docs")


def ingest_docs(connector, embedder, store, tracker, chunk_size, chunk_overlap):
    """Ingest README and /docs markdown files."""
    logger.info("\n--- GitHub Docs ---")
    docs = connector.fetch_docs()
    logger.info(f"Found {len(docs)} doc files across matching repos")

    total = 0
    skipped = 0
    errors = 0
    current_ids = set()

    for doc in docs:
        doc_id = f"{doc.repo_name}/{doc.file_path}"
        current_ids.add(doc_id)
        # Use content hash as fingerprint (docs don't have last_modified)
        fingerprint = str(hash(doc.content))

        if not tracker.has_changed("github_docs", doc_id, fingerprint):
            skipped += 1
            continue

        try:
            store.delete_by_filter(source_type="github_docs", repo_name=doc.repo_name, file_path=doc.file_path)

            chunks = chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                tracker.mark_ingested("github_docs", doc_id, fingerprint)
                continue

            texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed_batch(texts)

            payloads = [
                {
                    "source_type": "github_docs",
                    "repo_name": doc.repo_name,
                    "repo_url": doc.repo_url,
                    "file_path": doc.file_path,
                    "heading_context": chunk.heading_context,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": len(chunks),
                    "text": chunk.text,
                }
                for chunk, _ in zip(chunks, vectors)
            ]

            count = store.upsert_chunks(vectors, payloads)
            total += count
            tracker.mark_ingested("github_docs", doc_id, fingerprint)
            logger.info(f"  [{doc.repo_name}/{doc.file_path}] → {count} chunks")
        except Exception as e:
            errors += 1
            logger.error(f"  [{doc.repo_name}/{doc.file_path}] → ERROR: {e}")

    # Clean up deleted docs
    for old_id in tracker.get_all_keys("github_docs") - current_ids:
        parts = old_id.split("/", 1)
        if len(parts) == 2:
            store.delete_by_filter(source_type="github_docs", repo_name=parts[0], file_path=parts[1])
        tracker.remove("github_docs", old_id)

    return total, skipped, errors


def ingest_issues_and_prs(connector, embedder, store, tracker, chunk_size, chunk_overlap):
    """Ingest issues and merged PRs."""
    logger.info("\n--- GitHub Issues ---")
    issues = connector.fetch_issues()
    logger.info(f"Found {len(issues)} issues across matching repos")

    logger.info("\n--- GitHub PRs ---")
    prs = connector.fetch_merged_prs()
    logger.info(f"Found {len(prs)} merged PRs across matching repos")

    all_items = issues + prs
    total = 0
    skipped = 0
    errors = 0

    for item in all_items:
        item_id = f"{item.repo_name}/{item.source_type}/{item.number}"
        fingerprint = item.last_modified or str(hash(item.body))

        if not tracker.has_changed(item.source_type, item_id, fingerprint):
            skipped += 1
            continue

        try:
            number_key = "issue_number" if item.source_type == "github_issue" else "pr_number"
            store.delete_by_filter(source_type=item.source_type, repo_name=item.repo_name, **{number_key: item.number})

            chunks = chunk_text(item.body, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                tracker.mark_ingested(item.source_type, item_id, fingerprint)
                continue

            texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed_batch(texts)

            payloads = [
                {
                    "source_type": item.source_type,
                    "repo_name": item.repo_name,
                    "repo_url": item.repo_url,
                    number_key: item.number,
                    "title": item.title,
                    "author": item.author,
                    "last_modified": item.last_modified,
                    "heading_context": chunk.heading_context,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": len(chunks),
                    "text": chunk.text,
                }
                for chunk, _ in zip(chunks, vectors)
            ]

            count = store.upsert_chunks(vectors, payloads)
            total += count
            tracker.mark_ingested(item.source_type, item_id, fingerprint)
            label = f"#{item.number}" if item.source_type == "github_issue" else f"PR#{item.number}"
            logger.info(f"  [{item.repo_name} {label}] {item.title} → {count} chunks")
        except Exception as e:
            errors += 1
            logger.error(f"  [{item.repo_name} #{item.number}] → ERROR: {e}")

    return total, skipped, errors


def main():
    load_dotenv()
    start = time.time()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    github_token = os.environ.get("GITHUB_TOKEN")
    github_org = os.environ.get("GITHUB_ORG")
    if not all([github_token, github_org]):
        logger.error("Set GITHUB_TOKEN and GITHUB_ORG in .env")
        sys.exit(1)

    repo_prefix = config.get("github", {}).get("repo_prefix", "")
    lookback_months = config.get("github", {}).get("issue_lookback_months", 12)
    chunk_size = config.get("chunking", {}).get("chunk_size", 512)
    chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 50)
    embed_model = config["embedding"]["model"]
    collection_name = config["qdrant"]["collection_name"]
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    docs_connector = GitHubDocsConnector(github_token, github_org, repo_prefix)
    issues_connector = GitHubIssuesConnector(github_token, github_org, repo_prefix, lookback_months)
    embedder = Embedder(model=embed_model, ollama_url=ollama_url)
    store = VectorStore(collection_name=collection_name, host=qdrant_host, port=qdrant_port)
    tracker = IngestionTracker()

    store.ensure_collection(vector_size=embedder.vector_size())

    doc_chunks, doc_skipped, doc_errors = ingest_docs(
        docs_connector, embedder, store, tracker, chunk_size, chunk_overlap
    )
    issue_chunks, issue_skipped, issue_errors = ingest_issues_and_prs(
        issues_connector, embedder, store, tracker, chunk_size, chunk_overlap
    )

    tracker.save()
    elapsed = time.time() - start

    total_chunks = doc_chunks + issue_chunks
    total_skipped = doc_skipped + issue_skipped
    total_errors = doc_errors + issue_errors

    log_with_data(
        logger, logging.INFO,
        f"Done: {total_chunks} ingested, {total_skipped} skipped, {total_errors} errors ({elapsed:.1f}s)",
        source="github_docs", chunks=total_chunks,
        skipped=total_skipped, errors=total_errors, elapsed_seconds=round(elapsed, 1),
    )

    info = store.collection_info()
    logger.info(f"Collection: {info['name']} | Points: {info['points_count']} | Status: {info['status']}")


if __name__ == "__main__":
    main()
