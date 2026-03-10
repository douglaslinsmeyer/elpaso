"""Ingest Confluence pages — fetch, chunk, embed, store in Qdrant."""

import os
import sys

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.confluence import ConfluenceConnector
from pipeline.chunker import chunk_text
from pipeline.embedder import Embedder
from pipeline.store import VectorStore


def main():
    load_dotenv()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Validate env vars
    confluence_url = os.environ.get("CONFLUENCE_URL")
    confluence_user = os.environ.get("CONFLUENCE_USERNAME")
    confluence_token = os.environ.get("CONFLUENCE_API_TOKEN")
    if not all([confluence_url, confluence_user, confluence_token]):
        print("ERROR: Set CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN in .env")
        sys.exit(1)

    spaces = config.get("confluence", {}).get("spaces", [])
    chunk_size = config.get("chunking", {}).get("chunk_size", 512)
    chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 50)
    embed_model = config["embedding"]["model"]
    collection_name = config["qdrant"]["collection_name"]
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    connector = ConfluenceConnector(confluence_url, confluence_user, confluence_token)
    embedder = Embedder(model=embed_model, ollama_url=ollama_url)
    store = VectorStore(collection_name=collection_name, host=qdrant_host, port=qdrant_port)

    total_pages = 0
    total_chunks = 0

    for space_key in spaces:
        print(f"Fetching pages from space: {space_key}...")
        pages = connector.fetch_pages(space_key)
        print(f"  Found {len(pages)} pages with content")
        total_pages += len(pages)

        errors = 0
        for page in pages:
            try:
                chunks = chunk_text(page.body_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    continue

                texts = [chunk.text for chunk in chunks]
                vectors = embedder.embed_batch(texts)

                # Ensure collection exists (uses first vector to determine dimensions)
                if total_chunks == 0:
                    store.ensure_collection(vector_size=len(vectors[0]))

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
                print(f"  [{page.title}] → {count} chunks")
            except Exception as e:
                errors += 1
                print(f"  [{page.title}] → ERROR: {e}")

    print(f"\nDone: {total_pages} pages → {total_chunks} chunks stored in '{collection_name}'")
    if errors:
        print(f"  ({errors} pages failed — see errors above)")

    if total_chunks > 0:
        info = store.collection_info()
        print(f"Collection: {info['name']} | Points: {info['points_count']} | Status: {info['status']}")


if __name__ == "__main__":
    main()
