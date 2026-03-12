"""Tests for unified ingestion orchestrator."""

from unittest.mock import MagicMock, patch

from scripts.ingest_all import run_ingestion


class TestRunIngestion:
    def _make_mocks(self):
        config = {
            "confluence": {"spaces": ["ISS"]},
            "github": {"repo_prefix": "mes-", "issue_lookback_months": 12,
                       "code_extensions": [".cs"], "skip_patterns": []},
            "chunking": {"chunk_size": 512, "chunk_overlap": 50},
        }
        tracker = MagicMock()
        embedder = MagicMock()
        store = MagicMock()
        return config, tracker, embedder, store

    @patch("scripts.ingest_all.run_ingestion.__module__", "scripts.ingest_all")
    @patch("scripts.ingest_confluence.run_confluence_ingestion")
    @patch("scripts.ingest_github_docs.run_github_docs_ingestion")
    @patch("scripts.ingest_github_code.run_github_code_ingestion")
    def test_runs_all_sources(self, mock_code, mock_docs, mock_confluence):
        config, tracker, embedder, store = self._make_mocks()

        mock_confluence.return_value = {"pages": 10, "chunks": 50, "skipped": 2, "errors": 0}
        mock_docs.return_value = {"chunks": 30, "skipped": 5, "errors": 0}
        mock_code.return_value = {"files": 20, "chunks": 100, "skipped": 3, "errors": 0}

        stats = run_ingestion(
            ["confluence", "github_docs", "github_code"],
            config, tracker, embedder, store,
        )

        mock_confluence.assert_called_once_with(config, tracker, embedder, store)
        mock_docs.assert_called_once_with(config, tracker, embedder, store)
        mock_code.assert_called_once_with(config, tracker, embedder, store)

        assert stats["confluence"]["chunks"] == 50
        assert stats["github_docs"]["chunks"] == 30
        assert stats["github_code"]["chunks"] == 100
        assert stats["_errors_occurred"] is False

    @patch("scripts.ingest_confluence.run_confluence_ingestion")
    def test_runs_single_source(self, mock_confluence):
        config, tracker, embedder, store = self._make_mocks()
        mock_confluence.return_value = {"pages": 5, "chunks": 20, "skipped": 1, "errors": 0}

        stats = run_ingestion(["confluence"], config, tracker, embedder, store)

        mock_confluence.assert_called_once()
        assert "github_docs" not in stats
        assert "github_code" not in stats

    @patch("scripts.ingest_confluence.run_confluence_ingestion")
    def test_reports_errors(self, mock_confluence):
        config, tracker, embedder, store = self._make_mocks()
        mock_confluence.side_effect = RuntimeError("Missing credentials")

        stats = run_ingestion(["confluence"], config, tracker, embedder, store)

        assert "error" in stats["confluence"]
        assert stats["_errors_occurred"] is True

    @patch("scripts.ingest_github_code.run_github_code_ingestion")
    def test_nonzero_errors_flag(self, mock_code):
        config, tracker, embedder, store = self._make_mocks()
        mock_code.return_value = {"files": 10, "chunks": 40, "skipped": 0, "errors": 2}

        stats = run_ingestion(["github_code"], config, tracker, embedder, store)

        assert stats["_errors_occurred"] is True
