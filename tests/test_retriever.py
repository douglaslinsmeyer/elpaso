"""Tests for the retrieval and prompt formatting pipeline."""

from unittest.mock import MagicMock, patch

from mcp_server.prompts import SYSTEM_PROMPT, build_synthesis_prompt


class TestBuildSynthesisPrompt:
    def test_includes_question(self):
        prompt = build_synthesis_prompt("how does shipping work", [])
        assert "how does shipping work" in prompt

    def test_includes_confluence_chunk(self):
        chunks = [
            {
                "source_type": "confluence",
                "page_title": "Shipping Guide",
                "page_url": "https://example.com/shipping",
                "heading_context": "Overview",
                "text": "Shipping uses Logistyx TME.",
            }
        ]
        prompt = build_synthesis_prompt("how does shipping work", chunks)
        assert "[Source 1]" in prompt
        assert "confluence" in prompt
        assert "Shipping Guide" in prompt
        assert "Overview" in prompt
        assert "Shipping uses Logistyx TME." in prompt

    def test_includes_code_chunk(self):
        chunks = [
            {
                "source_type": "github_code",
                "repo_name": "mes-shipping",
                "file_path": "src/ShippingService.cs",
                "class_name": "ShippingService",
                "method_name": "ProcessShipment",
                "repo_url": "https://github.com/org/mes-shipping",
                "text": "public void ProcessShipment() {}",
            }
        ]
        prompt = build_synthesis_prompt("shipping code", chunks)
        assert "github_code" in prompt
        assert "ShippingService" in prompt
        assert "ProcessShipment" in prompt

    def test_multiple_chunks_numbered(self):
        chunks = [
            {"source_type": "confluence", "text": "Chunk one", "page_url": ""},
            {"source_type": "github_docs", "text": "Chunk two", "repo_url": ""},
        ]
        prompt = build_synthesis_prompt("test", chunks)
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt

    def test_empty_chunks(self):
        prompt = build_synthesis_prompt("test question", [])
        assert "test question" in prompt


class TestSystemPrompt:
    def test_contains_key_instructions(self):
        assert "ONLY" in SYSTEM_PROMPT
        assert "cite" in SYSTEM_PROMPT.lower()
        assert "never use outside knowledge" in SYSTEM_PROMPT.lower()


class TestRetriever:
    @patch("mcp_server.retriever.requests.post")
    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_ask_returns_llm_response(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls, mock_post):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:30b-a3b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"source_type": "confluence", "text": "Test content", "page_url": ""},
        ]
        mock_store_cls.return_value = mock_store

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "The answer is 42 [Source 1]."}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        answer = retriever.ask("what is the answer")

        assert "42" in answer
        assert "[Source 1]" in answer
        mock_embedder.embed.assert_called_once()
        mock_store.search.assert_called_once()
        mock_post.assert_called_once()

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_ask_no_results(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:30b-a3b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        answer = retriever.ask("something obscure")

        assert "No relevant information" in answer

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_scope_code_filters_source_types(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:30b-a3b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = []
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        retriever.ask("show me code", scope="code")

        # Code scope defaults to hybrid search mode
        mock_store.hybrid_search.assert_called_once()
        call_kwargs = mock_store.hybrid_search.call_args.kwargs
        assert call_kwargs["source_types"] == ["github_code"]
        assert call_kwargs["repo_name"] == ""
        assert call_kwargs["space_key"] == ""

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_search_returns_raw_chunks(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:30b-a3b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        chunks = [
            {"source_type": "github_code", "text": "public void Ship() {}", "score": 0.92, "repo_name": "mes-shipping"},
            {"source_type": "confluence", "text": "Shipping overview", "score": 0.85, "page_url": "https://example.com"},
        ]
        mock_store = MagicMock()
        mock_store.search.return_value = chunks
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        results = retriever.search("shipping", top_k=5)

        assert len(results) == 2
        assert results[0]["score"] == 0.92
        assert results[0]["source_type"] == "github_code"
        assert results[1]["text"] == "Shipping overview"

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_search_respects_top_k_override(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:30b-a3b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        retriever.search("test", top_k=3)

        mock_store.search.assert_called_once_with(
            mock_embedder.embed.return_value,
            top_k=7,  # 3 + 4 extra for dedup
            source_types=None,
            repo_name="",
            space_key="",
        )

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_scope_docs_excludes_issues(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:30b-a3b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        retriever.search("docs query", scope="docs")

        call_args = mock_store.search.call_args
        source_types = call_args.kwargs.get("source_types") or call_args[1].get("source_types")
        assert "github_issue" not in source_types
        assert "github_pr" not in source_types
        assert "confluence" in source_types
        assert "github_docs" in source_types

    @patch("mcp_server.retriever.VectorStore")
    @patch("mcp_server.retriever.Embedder")
    @patch("mcp_server.retriever.yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_scope_issues_filters_correctly(self, mock_open, mock_yaml, mock_embedder_cls, mock_store_cls):
        mock_yaml.return_value = {
            "embedding": {"model": "nomic-embed-text"},
            "llm": {"model": "qwen3:30b-a3b"},
            "qdrant": {"collection_name": "el_paso"},
            "retrieval": {"top_k": 5},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_store_cls.return_value = mock_store

        from mcp_server.retriever import Retriever
        retriever = Retriever()
        retriever.search("bug query", scope="issues")

        call_args = mock_store.search.call_args
        source_types = call_args.kwargs.get("source_types") or call_args[1].get("source_types")
        assert source_types == ["github_issue", "github_pr"]
