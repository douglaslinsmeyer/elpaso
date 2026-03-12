"""Unified ingestion script — runs all sources with shared config/connections."""

import argparse
import logging
import os
import sys
import time

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embedder import Embedder
from pipeline.ingestion_tracker import IngestionTracker
from pipeline.logger import get_logger, log_with_data
from pipeline.store import VectorStore

logger = get_logger("el_paso.ingest.all")

VALID_SOURCES = ("confluence", "github_docs", "github_code")


def run_ingestion(sources: list[str], config: dict, tracker: IngestionTracker, embedder: Embedder, store: VectorStore) -> dict:
    """Run ingestion for the specified sources. Returns aggregated stats."""
    all_stats: dict[str, dict] = {}
    errors_occurred = False

    for source in sources:
        logger.info(f"\n{'='*60}\nIngesting: {source}\n{'='*60}")
        try:
            if source == "confluence":
                from scripts.ingest_confluence import run_confluence_ingestion
                stats = run_confluence_ingestion(config, tracker, embedder, store)
            elif source == "github_docs":
                from scripts.ingest_github_docs import run_github_docs_ingestion
                stats = run_github_docs_ingestion(config, tracker, embedder, store)
            elif source == "github_code":
                from scripts.ingest_github_code import run_github_code_ingestion
                stats = run_github_code_ingestion(config, tracker, embedder, store)
            else:
                logger.error(f"Unknown source: {source}")
                continue

            all_stats[source] = stats
            if stats.get("errors", 0) > 0:
                errors_occurred = True

            log_with_data(
                logger, logging.INFO,
                f"Completed {source}: {stats}",
                source=source, **stats,
            )
        except Exception as e:
            errors_occurred = True
            all_stats[source] = {"error": str(e)}
            logger.error(f"Failed to ingest {source}: {e}")

    all_stats["_errors_occurred"] = errors_occurred
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Unified El Paso ingestion")
    parser.add_argument(
        "--source",
        choices=VALID_SOURCES,
        help="Run only a specific source (default: all)",
    )
    args = parser.parse_args()

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

    sources = [args.source] if args.source else list(VALID_SOURCES)
    all_stats = run_ingestion(sources, config, tracker, embedder, store)

    tracker.save()
    elapsed = time.time() - start

    total_chunks = sum(s.get("chunks", 0) for s in all_stats.values() if isinstance(s, dict) and "chunks" in s)
    total_skipped = sum(s.get("skipped", 0) for s in all_stats.values() if isinstance(s, dict) and "skipped" in s)
    total_errors = sum(s.get("errors", 0) for s in all_stats.values() if isinstance(s, dict) and "errors" in s)

    log_with_data(
        logger, logging.INFO,
        f"All done: {total_chunks} ingested, {total_skipped} skipped, {total_errors} errors ({elapsed:.1f}s)",
        sources=sources, chunks=total_chunks, skipped=total_skipped,
        errors=total_errors, elapsed_seconds=round(elapsed, 1),
    )

    try:
        info = store.collection_info()
        logger.info(f"Collection: {info['name']} | Points: {info['points_count']} | Status: {info['status']}")
    except Exception:
        pass

    if all_stats.get("_errors_occurred"):
        sys.exit(1)


if __name__ == "__main__":
    main()
