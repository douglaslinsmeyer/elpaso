"""Qdrant vector store — collection management, upsert, and hybrid search."""

import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchAny, MatchText, MatchValue,
    PayloadSchemaType, PointStruct, TextIndexParams, TokenizerType, VectorParams,
)


def split_identifiers(text: str) -> str:
    """Split PascalCase/camelCase identifiers into separate words.

    'ProcessDeadLetterMessage' → 'Process Dead Letter Message ProcessDeadLetterMessage'

    Preserves the original text and appends split tokens so both exact
    and word-level matching work.
    """
    # Find PascalCase/camelCase words (2+ chars with internal case boundaries)
    identifier_pattern = re.compile(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:[A-Z][a-z]+)+')

    identifiers = identifier_pattern.findall(text)
    if not identifiers:
        return text

    expanded_parts = []
    for ident in identifiers:
        # Split on case boundaries: 'ProcessDeadLetter' → ['Process', 'Dead', 'Letter']
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', ident)
        words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', words)
        expanded_parts.append(words)

    return text + " " + " ".join(expanded_parts)


def _chunk_key(chunk: dict) -> str:
    """Generate a dedup key for a chunk based on its text content."""
    return chunk.get("text", "")[:200]


def reciprocal_rank_fusion(
    *result_lists: list[dict],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (rrf_k + rank)) across all lists where the item appears.
    Higher rrf_k gives more weight to items appearing in multiple lists vs. rank position.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, 1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**chunk_map[key], "rrf_score": score} for key, score in ranked]


class VectorStore:
    """Manages a Qdrant collection for El Paso document chunks."""

    def __init__(
        self,
        collection_name: str = "el_paso",
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it doesn't exist, then ensure payload indexes."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for hybrid search and filtering."""
        # Full-text index on text field for keyword search
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=40,
                lowercase=True,
            ),
        )
        # Keyword indexes for code metadata filtering
        for field in ("class_name", "method_name", "namespace", "file_path", "implements_interfaces", "repo_name"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        # Keyword indexes for community context management
        for field in ("identifier", "tags"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(
        self,
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> int:
        """Upsert embedded chunks with metadata payloads. Returns count upserted."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            for vector, payload in zip(vectors, payloads)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def delete_by_filter(self, **field_matches) -> None:
        """Delete points matching exact field values."""
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in field_matches.items()
        ]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=conditions),
        )

    def delete_collection(self) -> None:
        """Drop the entire collection."""
        self.client.delete_collection(self.collection_name)

    def search(
        self, query_vector: list[float], top_k: int = 5,
        source_types: list[str] | None = None,
        repo_name: str = "",
        space_key: str = "",
    ) -> list[dict]:
        """Search for similar vectors with optional filters."""
        conditions = []
        if source_types:
            conditions.append(
                FieldCondition(key="source_type", match=MatchAny(any=source_types))
            )
        if repo_name:
            conditions.append(
                FieldCondition(key="repo_name", match=MatchValue(value=repo_name))
            )
        if space_key:
            conditions.append(
                FieldCondition(key="space_key", match=MatchValue(value=space_key))
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        return [
            {
                "score": point.score,
                **point.payload,
            }
            for point in results.points
        ]

    def keyword_search(
        self, query_text: str, top_k: int = 5,
        source_types: list[str] | None = None,
        repo_name: str = "",
        space_key: str = "",
    ) -> list[dict]:
        """Search using full-text match on text and keyword match on code metadata.

        Runs a text match on the `text` field (with PascalCase splitting) and
        also checks `method_name`, `class_name`, `namespace`, and `file_path`
        keyword indexes. Results from any matching strategy are combined.
        """
        shared_conditions = []
        if source_types:
            shared_conditions.append(
                FieldCondition(key="source_type", match=MatchAny(any=source_types))
            )
        if repo_name:
            shared_conditions.append(
                FieldCondition(key="repo_name", match=MatchValue(value=repo_name))
            )
        if space_key:
            shared_conditions.append(
                FieldCondition(key="space_key", match=MatchValue(value=space_key))
            )

        # Strategy 1: full-text search on text field (with identifier splitting)
        expanded_query = split_identifiers(query_text)
        text_conditions = [
            FieldCondition(key="text", match=MatchText(text=expanded_query)),
            *shared_conditions,
        ]

        # Strategy 2: text match on code metadata fields
        metadata_fields = ["method_name", "class_name", "namespace", "file_path"]
        metadata_or_conditions = [
            FieldCondition(key=field, match=MatchText(text=query_text))
            for field in metadata_fields
        ]

        all_results: dict[str, dict] = {}

        # Run text search
        text_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=text_conditions),
            limit=top_k,
            with_payload=True,
        )
        for point in text_results[0]:
            key = point.payload.get("text", "")[:200]
            if key not in all_results:
                all_results[key] = {"score": 0.0, **point.payload}

        # Run metadata search (any field matches)
        if metadata_or_conditions:
            meta_filter = Filter(
                should=metadata_or_conditions,
                must=shared_conditions if shared_conditions else None,
            )
            meta_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=meta_filter,
                limit=top_k,
                with_payload=True,
            )
            for point in meta_results[0]:
                key = point.payload.get("text", "")[:200]
                if key not in all_results:
                    all_results[key] = {"score": 0.0, **point.payload}

        return list(all_results.values())[:top_k]

    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int = 5, rrf_k: int = 60,
        source_types: list[str] | None = None,
        repo_name: str = "",
        space_key: str = "",
    ) -> list[dict]:
        """Combine semantic and keyword search using reciprocal rank fusion (RRF)."""
        # Fetch more from each to have good candidates for fusion
        fetch_k = top_k * 2

        semantic_results = self.search(
            query_vector, top_k=fetch_k,
            source_types=source_types, repo_name=repo_name, space_key=space_key,
        )
        keyword_results = self.keyword_search(
            query_text, top_k=fetch_k,
            source_types=source_types, repo_name=repo_name, space_key=space_key,
        )

        return reciprocal_rank_fusion(semantic_results, keyword_results, top_k=top_k, rrf_k=rrf_k)

    def scroll_by_filter(self, limit: int = 100, **field_matches) -> list[dict]:
        """Scroll through points matching filter criteria. Returns payloads."""
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in field_matches.items()
        ]
        scroll_filter = Filter(must=conditions) if conditions else None
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )
        return [point.payload for point in points]

    def collection_info(self) -> dict:
        """Return collection point count and status."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status.value,
        }
