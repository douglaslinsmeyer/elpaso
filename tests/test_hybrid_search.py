"""Tests for hybrid search and reciprocal rank fusion."""

from unittest.mock import MagicMock, patch

from pipeline.store import reciprocal_rank_fusion, split_identifiers


class TestSplitIdentifiers:
    def test_pascal_case(self):
        result = split_identifiers("ProcessDeadLetterMessage")
        assert "Process" in result
        assert "Dead" in result
        assert "Letter" in result
        assert "Message" in result
        assert "ProcessDeadLetterMessage" in result  # original preserved

    def test_camel_case(self):
        result = split_identifiers("processDeadLetterMessage")
        assert "process" in result
        assert "Dead" in result
        assert "Letter" in result

    def test_no_identifiers(self):
        result = split_identifiers("hello world")
        assert result == "hello world"

    def test_mixed_text(self):
        result = split_identifiers("find the ProcessShipment method")
        assert "Process" in result
        assert "Shipment" in result
        assert "find the" in result

    def test_acronyms(self):
        result = split_identifiers("IRabbitMQManager")
        assert "IRabbitMQManager" in result


class TestReciprocalRankFusion:
    def test_merges_two_lists(self):
        semantic = [
            {"text": "chunk A", "score": 0.9},
            {"text": "chunk B", "score": 0.8},
            {"text": "chunk C", "score": 0.7},
        ]
        keyword = [
            {"text": "chunk B", "score": 0.0},
            {"text": "chunk D", "score": 0.0},
            {"text": "chunk A", "score": 0.0},
        ]

        results = reciprocal_rank_fusion(semantic, keyword, top_k=4, rrf_k=60)

        texts = [r["text"] for r in results]
        # B appears in both lists (rank 2 semantic + rank 1 keyword) → highest RRF
        # A appears in both (rank 1 semantic + rank 3 keyword) → second highest
        assert texts[0] == "chunk B"
        assert texts[1] == "chunk A"
        assert len(results) == 4

    def test_rrf_scores_present(self):
        results = reciprocal_rank_fusion(
            [{"text": "a", "score": 0.9}],
            [{"text": "a", "score": 0.0}],
            top_k=1,
        )
        assert "rrf_score" in results[0]
        assert results[0]["rrf_score"] > 0

    def test_respects_top_k(self):
        items = [{"text": f"chunk {i}", "score": 0.5} for i in range(10)]
        results = reciprocal_rank_fusion(items, top_k=3)
        assert len(results) == 3

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], [], top_k=5)
        assert results == []

    def test_single_list(self):
        items = [{"text": "only", "score": 0.9}]
        results = reciprocal_rank_fusion(items, top_k=5)
        assert len(results) == 1
        assert results[0]["text"] == "only"

    def test_item_in_one_list_only(self):
        semantic = [{"text": "unique_semantic", "score": 0.9}]
        keyword = [{"text": "unique_keyword", "score": 0.0}]

        results = reciprocal_rank_fusion(semantic, keyword, top_k=2, rrf_k=60)
        texts = [r["text"] for r in results]
        assert "unique_semantic" in texts
        assert "unique_keyword" in texts


class TestRetrieverHybridMode:
    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_code_scope_defaults_to_hybrid(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:8b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
            "search": {"default_mode": "semantic", "code_default_mode": "hybrid", "rrf_k": 60},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = [
            {"text": "public void Ship() {}", "score": 0.9, "rrf_score": 0.03},
        ]
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        results = retriever.search("ProcessShipment", scope="code")

        mock_store.hybrid_search.assert_called_once()
        assert len(results) == 1

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_docs_scope_defaults_to_semantic(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:8b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
            "search": {"default_mode": "semantic", "code_default_mode": "hybrid", "rrf_k": 60},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        retriever.search("shipping docs", scope="docs")

        mock_store.search.assert_called_once()
        mock_store.hybrid_search.assert_not_called()

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_explicit_mode_overrides_default(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:8b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
            "search": {"default_mode": "semantic", "code_default_mode": "hybrid", "rrf_k": 60},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.keyword_search.return_value = []
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        retriever.search("ProcessShipment", scope="code", mode="keyword")

        mock_store.keyword_search.assert_called_once()
        mock_store.hybrid_search.assert_not_called()
