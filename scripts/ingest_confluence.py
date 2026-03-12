"""Ingest Confluence pages — fetch, chunk, embed, store in Qdrant."""

import logging
import os
import sys
import time

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.confluence import ConfluenceConnector
from pipeline.chunker import chunk_text
from pipeline.embedder import Embedder
from pipeline.ingestion_tracker import IngestionTracker
from pipeline.logger import get_logger, log_with_data
from pipeline.store import VectorStore

logger = get_logger("el_paso.ingest.confluence")


def run_confluence_ingestion(config: dict, tracker: IngestionTracker, embedder: Embedder, store: VectorStore) -> dict:
    """Run Confluence ingestion. Returns stats dict with pages, chunks, skipped, errors."""
    confluence_url = os.environ.get("CONFLUENCE_URL")
    confluence_user = os.environ.get("CONFLUENCE_USERNAME")
    confluence_token = os.environ.get("CONFLUENCE_API_TOKEN")
    if not all([confluence_url, confluence_user, confluence_token]):
        raise RuntimeError("Set CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN in .env")

    spaces = config.get("confluence", {}).get("spaces", [])
    chunk_size = config.get("chunking", {}).get("chunk_size", 512)
    chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 50)

    connector = ConfluenceConnector(confluence_url, confluence_user, confluence_token)

    total_pages = 0
    total_chunks = 0
    skipped = 0
    errors = 0

    for space_key in spaces:
        logger.info(f"Fetching pages from space: {space_key}...")
        pages = connector.fetch_pages(space_key)
        logger.info(f"  Found {len(pages)} pages with content")
        total_pages += len(pages)

        current_ids = set()
        for page in pages:
            current_ids.add(page.page_id)

            if not tracker.has_changed("confluence", page.page_id, page.last_modified):
                skipped += 1
                continue

            try:
                store.delete_by_filter(source_type="confluence", page_id=page.page_id)

                chunks = chunk_text(page.body_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    tracker.mark_ingested("confluence", page.page_id, page.last_modified)
                    continue

                texts = [chunk.text for chunk in chunks]
                vectors = embedder.embed_batch(texts)

                payloads = [
                    {
                        "source_type": "confluence",
                        "space_key": page.space_key,
                        "page_id": page.page_id,
                        "page_title": page.title,
                        "page_url": page.url,
                        "author": page.author,
                        "last_modified": page.last_modified,
                        "heading_context": chunk.heading_context,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": len(chunks),
                        "text": chunk.text,
                    }
                    for chunk, _ in zip(chunks, vectors)
                ]

                count = store.upsert_chunks(vectors, payloads)
                total_chunks += count
                tracker.mark_ingested("confluence", page.page_id, page.last_modified)
                logger.info(f"  [{page.title}] → {count} chunks")
            except Exception as e:
                errors += 1
                logger.error(f"  [{page.title}] → ERROR: {e}")

        tracked_ids = tracker.get_all_keys("confluence")
        for old_id in tracked_ids - current_ids:
            store.delete_by_filter(source_type="confluence", page_id=old_id)
            tracker.remove("confluence", old_id)

    return {"pages": total_pages, "chunks": total_chunks, "skipped": skipped, "errors": errors}


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

    stats = run_confluence_ingestion(config, tracker, embedder, store)

    tracker.save()
    elapsed = time.time() - start

    log_with_data(
        logger, logging.INFO,
        f"Done: {stats['pages']} pages → {stats['chunks']} ingested, {stats['skipped']} skipped, {stats['errors']} errors ({elapsed:.1f}s)",
        source="confluence", **stats, elapsed_seconds=round(elapsed, 1),
    )

    try:
        info = store.collection_info()
        logger.info(f"Collection: {info['name']} | Points: {info['points_count']} | Status: {info['status']}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
