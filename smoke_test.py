"""Smoke test — validates Ollama + Qdrant + embeddings round-trip."""

import sys
import uuid

import requests
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def check_ollama_health(ollama_url: str, llm_model: str) -> bool:
    """Check Ollama is running and the LLM model is available."""
    print("1. Ollama health check...", end=" ")
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # Match on model family (e.g. "qwen3:30b-a3b" matches "qwen3:30b-a3b")
        if not any(llm_model in m for m in models):
            print(f"FAIL — model '{llm_model}' not found. Available: {models}")
            return False
        print("PASS")
        return True
    except requests.ConnectionError:
        print("FAIL — cannot connect to Ollama. Is it running?")
        return False
    except Exception as e:
        print(f"FAIL — {e}")
        return False


def check_embedding(ollama_url: str, embed_model: str) -> list[float] | None:
    """Embed a test sentence and verify vector dimensions."""
    print("2. Embedding test...", end=" ")
    try:
        resp = requests.post(
            f"{ollama_url}/api/embed",
            json={"model": embed_model, "input": "El Paso smoke test sentence"},
            timeout=30,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if not embeddings:
            print("FAIL — no embeddings returned")
            return None
        vector = embeddings[0]
        print(f"PASS — dimension={len(vector)}")
        return vector
    except requests.ConnectionError:
        print("FAIL — cannot connect to Ollama")
        return None
    except Exception as e:
        print(f"FAIL — {e}")
        return None


def check_qdrant_health(qdrant_host: str, qdrant_port: int) -> QdrantClient | None:
    """Connect to Qdrant and verify it responds."""
    print("3. Qdrant health check...", end=" ")
    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5)
        client.get_collections()
        print("PASS")
        return client
    except Exception as e:
        print(f"FAIL — {e}")
        return None


def check_round_trip(
    client: QdrantClient, vector: list[float], collection_name: str
) -> bool:
    """Embed → upsert → query → verify match."""
    print("4. Round-trip test...", end=" ")
    test_collection = f"{collection_name}_smoke_test"
    try:
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE),
        )
        point_id = str(uuid.uuid4())
        client.upsert(
            collection_name=test_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": "El Paso smoke test sentence"},
                )
            ],
        )
        results = client.query_points(
            collection_name=test_collection,
            query=vector,
            limit=1,
        )
        if not results.points:
            print("FAIL — no results returned from query")
            return False
        if results.points[0].id != point_id:
            print("FAIL — returned point ID doesn't match")
            return False
        print("PASS")
        return True
    except Exception as e:
        print(f"FAIL — {e}")
        return False
    finally:
        print("5. Cleanup...", end=" ")
        try:
            client.delete_collection(collection_name=test_collection)
            print("PASS")
        except Exception as e:
            print(f"FAIL — {e}")


def main():
    config = load_config()
    ollama_url = "http://localhost:11434"
    llm_model = config["llm"]["model"]
    embed_model = config["embedding"]["model"]
    qdrant_host = "localhost"
    qdrant_port = 6333
    collection_name = config["qdrant"]["collection_name"]

    print("=" * 50)
    print("El Paso Smoke Test")
    print("=" * 50)

    failures = 0

    if not check_ollama_health(ollama_url, llm_model):
        failures += 1

    vector = check_embedding(ollama_url, embed_model)
    if vector is None:
        failures += 1

    client = check_qdrant_health(qdrant_host, qdrant_port)
    if client is None:
        failures += 1

    if client and vector:
        if not check_round_trip(client, vector, collection_name):
            failures += 1
    elif not client:
        print("4. Round-trip test... SKIP — Qdrant not available")
        print("5. Cleanup... SKIP")

    print("=" * 50)
    if failures:
        print(f"DONE — {failures} step(s) failed")
        sys.exit(1)
    else:
        print("DONE — all steps passed")


if __name__ == "__main__":
    main()
