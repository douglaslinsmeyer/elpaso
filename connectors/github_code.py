"""GitHub code connector — fetches source files from repos."""

import base64
import fnmatch
import time
from dataclasses import dataclass

from github import Github, GithubException


@dataclass
class GitHubCodeFile:
    repo_name: str
    repo_url: str
    file_path: str
    content: str
    language: str


class GitHubCodeConnector:
    """Fetches source code files from GitHub repos."""

    def __init__(
        self,
        token: str,
        org: str,
        repo_prefix: str = "",
        extensions: list[str] | None = None,
        skip_patterns: list[str] | None = None,
    ):
        self.github = Github(token)
        self.org = org
        self.repo_prefix = repo_prefix
        self.extensions = extensions or [".cs"]
        self.skip_patterns = skip_patterns or []

    EXTENSION_TO_LANGUAGE = {
        ".cs": "csharp",
        ".java": "java",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".py": "python",
        ".js": "javascript",
    }

    def _get_repos(self):
        """List repos matching the prefix filter."""
        org = self.github.get_organization(self.org)
        for repo in org.get_repos():
            if self.repo_prefix and not repo.name.startswith(self.repo_prefix):
                continue
            yield repo

    def _should_skip(self, path: str) -> bool:
        """Check if a file path matches any skip pattern."""
        parts = path.split("/")
        for pattern in self.skip_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
            # For ** patterns, check if any path component matches the dir name
            if "**" in pattern:
                dir_name = pattern.replace("**/", "").replace("/**", "")
                if dir_name in parts:
                    return True
            # For simple glob patterns, check against the filename
            if not "/" in pattern:
                if fnmatch.fnmatch(parts[-1], pattern):
                    return True
        return False

    def _get_language(self, path: str) -> str:
        """Determine language from file extension."""
        for ext, lang in self.EXTENSION_TO_LANGUAGE.items():
            if path.endswith(ext):
                return lang
        return "unknown"

    def _fetch_blob_with_retry(self, repo, sha: str, max_retries: int = 3):
        """Fetch a git blob with exponential backoff on transient errors."""
        for attempt in range(max_retries):
            try:
                return repo.get_git_blob(sha)
            except Exception:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None

    def _walk_tree(self, repo) -> list[GitHubCodeFile]:
        """Recursively walk repo tree and collect matching source files."""
        files: list[GitHubCodeFile] = []
        try:
            tree = repo.get_git_tree("HEAD", recursive=True)
        except GithubException:
            return files

        for item in tree.tree:
            if item.type != "blob":
                continue

            # Check extension
            has_ext = any(item.path.endswith(ext) for ext in self.extensions)
            if not has_ext:
                continue

            # Check skip patterns
            if self._should_skip(item.path):
                continue

            try:
                blob = self._fetch_blob_with_retry(repo, item.sha)
                if blob is None:
                    continue
                if blob.encoding == "base64":
                    content = base64.b64decode(blob.content).decode("utf-8", errors="replace")
                else:
                    content = blob.content or ""

                if content.strip():
                    files.append(GitHubCodeFile(
                        repo_name=repo.name,
                        repo_url=repo.html_url,
                        file_path=item.path,
                        content=content,
                        language=self._get_language(item.path),
                    ))
            except Exception:
                continue

        return files

    def fetch_code(self) -> list[GitHubCodeFile]:
        """Fetch all matching source files from all repos."""
        all_files: list[GitHubCodeFile] = []
        for repo in self._get_repos():
            files = self._walk_tree(repo)
            if files:
                all_files.extend(files)
        return all_files
