"""Confluence REST API connector — fetches pages from a space with pagination."""

from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup


@dataclass
class ConfluencePage:
    page_id: str
    title: str
    space_key: str
    url: str
    author: str
    last_modified: str
    body_text: str


class ConfluenceConnector:
    """Fetches pages from Confluence Cloud via REST API v2."""

    def __init__(self, base_url: str, username: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (username, api_token)
        self.session.headers.update({"Accept": "application/json"})

    def _html_to_text(self, html: str) -> str:
        """Convert Confluence HTML body to clean text preserving structure."""
        soup = BeautifulSoup(html, "html.parser")

        # Replace heading tags with markdown-style markers
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(tag.name[1])
            prefix = "#" * level
            tag.replace_with(f"\n{prefix} {tag.get_text(strip=True)}\n")

        # Replace list items with bullets
        for tag in soup.find_all("li"):
            tag.replace_with(f"\n- {tag.get_text(strip=True)}")

        # Replace table cells with pipe-separated values
        for tag in soup.find_all("td"):
            tag.replace_with(f" | {tag.get_text(strip=True)}")
        for tag in soup.find_all("th"):
            tag.replace_with(f" | {tag.get_text(strip=True)}")
        for tag in soup.find_all("tr"):
            tag.replace_with(f"\n{tag.get_text()}")

        text = soup.get_text()

        # Collapse multiple blank lines
        lines = []
        prev_blank = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if not prev_blank:
                    lines.append("")
                prev_blank = True
            else:
                lines.append(stripped)
                prev_blank = False

        return "\n".join(lines).strip()

    def fetch_pages(self, space_key: str) -> list[ConfluencePage]:
        """Fetch all pages in a space with their body content."""
        pages: list[ConfluencePage] = []
        url = f"{self.base_url}/rest/api/content"
        params = {
            "spaceKey": space_key,
            "type": "page",
            "expand": "body.storage,version,history",
            "limit": 25,
            "start": 0,
        }

        while True:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for result in data.get("results", []):
                body_html = result.get("body", {}).get("storage", {}).get("value", "")
                body_text = self._html_to_text(body_html)

                if not body_text.strip():
                    continue

                version = result.get("version", {})
                history = result.get("history", {})

                page = ConfluencePage(
                    page_id=result["id"],
                    title=result["title"],
                    space_key=space_key,
                    url=f"{self.base_url}{result.get('_links', {}).get('webui', '')}",
                    author=history.get("createdBy", {}).get("displayName", "unknown"),
                    last_modified=version.get("when", datetime.now().isoformat()),
                    body_text=body_text,
                )
                pages.append(page)

            # Handle pagination
            next_link = data.get("_links", {}).get("next")
            if not next_link:
                break
            url = f"{self.base_url}{next_link}"
            params = {}  # next link includes params

        return pages
