"""Tests for the Confluence connector with mocked API responses."""

from unittest.mock import MagicMock, patch

from connectors.confluence import ConfluenceConnector


def _make_api_response(results, has_next=False):
    """Build a mock Confluence API response."""
    links = {}
    if has_next:
        links["next"] = "/rest/api/content?start=25"
    return {
        "results": results,
        "_links": links,
    }


def _make_page(page_id="123", title="Test Page", body="<p>Hello world</p>"):
    return {
        "id": page_id,
        "title": title,
        "body": {"storage": {"value": body}},
        "version": {"when": "2025-01-15T10:00:00Z"},
        "history": {"createdBy": {"displayName": "Test User"}},
        "_links": {"webui": f"/spaces/ISS/pages/{page_id}"},
    }


class TestHtmlToText:
    def setup_method(self):
        self.connector = ConfluenceConnector(
            "https://test.atlassian.net/wiki", "user", "token"
        )

    def test_plain_paragraph(self):
        result = self.connector._html_to_text("<p>Simple text.</p>")
        assert "Simple text." in result

    def test_heading_extraction(self):
        html = "<h1>Title</h1><p>Body text.</p>"
        result = self.connector._html_to_text(html)
        assert "# Title" in result
        assert "Body text." in result

    def test_nested_headings(self):
        html = "<h2>Subtitle</h2><p>Content</p>"
        result = self.connector._html_to_text(html)
        assert "## Subtitle" in result

    def test_list_extraction(self):
        html = "<ul><li>Item one</li><li>Item two</li></ul>"
        result = self.connector._html_to_text(html)
        assert "- Item one" in result
        assert "- Item two" in result

    def test_empty_html(self):
        result = self.connector._html_to_text("")
        assert result == ""

    def test_collapses_blank_lines(self):
        html = "<p>First</p><br/><br/><br/><p>Second</p>"
        result = self.connector._html_to_text(html)
        # Should not have more than one consecutive blank line
        assert "\n\n\n" not in result


class TestFetchPages:
    def setup_method(self):
        self.connector = ConfluenceConnector(
            "https://test.atlassian.net/wiki", "user", "token"
        )

    @patch.object(ConfluenceConnector, "__init__", lambda self, *a, **kw: None)
    def _make_connector_with_mock_session(self, responses):
        connector = ConfluenceConnector.__new__(ConfluenceConnector)
        connector.base_url = "https://test.atlassian.net/wiki"
        connector.session = MagicMock()
        mock_responses = []
        for resp_data in responses:
            mock_resp = MagicMock()
            mock_resp.json.return_value = resp_data
            mock_resp.raise_for_status = MagicMock()
            mock_responses.append(mock_resp)
        connector.session.get = MagicMock(side_effect=mock_responses)
        return connector

    def test_fetches_single_page(self):
        api_response = _make_api_response([_make_page()])
        connector = self._make_connector_with_mock_session([api_response])
        pages = connector.fetch_pages("ISS")
        assert len(pages) == 1
        assert pages[0].title == "Test Page"
        assert pages[0].space_key == "ISS"
        assert pages[0].page_id == "123"

    def test_skips_empty_pages(self):
        empty_page = _make_page(body="<p>   </p>")
        api_response = _make_api_response([empty_page])
        connector = self._make_connector_with_mock_session([api_response])
        pages = connector.fetch_pages("ISS")
        assert len(pages) == 0

    def test_handles_pagination(self):
        page1 = _make_page(page_id="1", title="Page One")
        page2 = _make_page(page_id="2", title="Page Two")
        resp1 = _make_api_response([page1], has_next=True)
        resp2 = _make_api_response([page2], has_next=False)
        connector = self._make_connector_with_mock_session([resp1, resp2])
        pages = connector.fetch_pages("ISS")
        assert len(pages) == 2
        assert pages[0].title == "Page One"
        assert pages[1].title == "Page Two"

    def test_extracts_metadata(self):
        api_response = _make_api_response([_make_page()])
        connector = self._make_connector_with_mock_session([api_response])
        pages = connector.fetch_pages("ISS")
        page = pages[0]
        assert page.author == "Test User"
        assert page.last_modified == "2025-01-15T10:00:00Z"
        assert "/spaces/ISS/pages/123" in page.url
