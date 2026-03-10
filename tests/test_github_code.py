"""Tests for GitHub code connector with mocked API responses."""

import base64
from unittest.mock import MagicMock

from github import GithubException

from connectors.github_code import GitHubCodeConnector


def _make_mock_repo(name="mes-test-service", html_url="https://github.com/org/mes-test-service"):
    repo = MagicMock()
    repo.name = name
    repo.html_url = html_url
    return repo


def _make_tree_item(path, item_type="blob", sha="abc123"):
    item = MagicMock()
    item.path = path
    item.type = item_type
    item.sha = sha
    return item


def _make_blob(content_str, encoding="base64"):
    blob = MagicMock()
    blob.encoding = encoding
    blob.content = base64.b64encode(content_str.encode()).decode() if encoding == "base64" else content_str
    return blob


class TestGitHubCodeConnector:
    def _make_connector(self, extensions=None, skip_patterns=None):
        connector = GitHubCodeConnector.__new__(GitHubCodeConnector)
        connector.github = MagicMock()
        connector.org = "test-org"
        connector.repo_prefix = "mes-"
        connector.extensions = extensions or [".cs"]
        connector.skip_patterns = skip_patterns or []
        return connector

    def test_filters_by_extension(self):
        connector = self._make_connector()
        repo = _make_mock_repo()
        tree = MagicMock()
        tree.tree = [
            _make_tree_item("src/Foo.cs"),
            _make_tree_item("src/bar.py"),
            _make_tree_item("README.md"),
        ]
        repo.get_git_tree.return_value = tree
        repo.get_git_blob.return_value = _make_blob("public class Foo {}")

        org_mock = MagicMock()
        org_mock.get_repos.return_value = [repo]
        connector.github.get_organization.return_value = org_mock

        files = connector.fetch_code()
        assert len(files) == 1
        assert files[0].file_path == "src/Foo.cs"
        assert files[0].language == "csharp"

    def test_skips_obj_bin(self):
        connector = self._make_connector(
            skip_patterns=["**/obj/**", "**/bin/**"]
        )
        repo = _make_mock_repo()
        tree = MagicMock()
        tree.tree = [
            _make_tree_item("src/Foo.cs"),
            _make_tree_item("obj/Debug/Foo.cs"),
            _make_tree_item("bin/Release/Bar.cs"),
        ]
        repo.get_git_tree.return_value = tree
        repo.get_git_blob.return_value = _make_blob("public class Foo {}")

        org_mock = MagicMock()
        org_mock.get_repos.return_value = [repo]
        connector.github.get_organization.return_value = org_mock

        files = connector.fetch_code()
        assert len(files) == 1
        assert files[0].file_path == "src/Foo.cs"

    def test_skips_designer_files(self):
        connector = self._make_connector(
            skip_patterns=["*.Designer.cs"]
        )
        repo = _make_mock_repo()
        tree = MagicMock()
        tree.tree = [
            _make_tree_item("src/Foo.cs"),
            _make_tree_item("src/Form1.Designer.cs"),
        ]
        repo.get_git_tree.return_value = tree
        repo.get_git_blob.return_value = _make_blob("public class Foo {}")

        org_mock = MagicMock()
        org_mock.get_repos.return_value = [repo]
        connector.github.get_organization.return_value = org_mock

        files = connector.fetch_code()
        assert len(files) == 1

    def test_filters_by_repo_prefix(self):
        connector = self._make_connector()
        matching = _make_mock_repo("mes-service")
        other = _make_mock_repo("other-service")

        org_mock = MagicMock()
        org_mock.get_repos.return_value = [matching, other]
        connector.github.get_organization.return_value = org_mock

        repos = list(connector._get_repos())
        assert len(repos) == 1
        assert repos[0].name == "mes-service"

    def test_skips_empty_files(self):
        connector = self._make_connector()
        repo = _make_mock_repo()
        tree = MagicMock()
        tree.tree = [_make_tree_item("src/Empty.cs")]
        repo.get_git_tree.return_value = tree
        repo.get_git_blob.return_value = _make_blob("   ")

        org_mock = MagicMock()
        org_mock.get_repos.return_value = [repo]
        connector.github.get_organization.return_value = org_mock

        files = connector.fetch_code()
        assert len(files) == 0

    def test_handles_repo_without_commits(self):
        connector = self._make_connector()
        repo = _make_mock_repo()
        repo.get_git_tree.side_effect = GithubException(409, "empty repo", None)

        org_mock = MagicMock()
        org_mock.get_repos.return_value = [repo]
        connector.github.get_organization.return_value = org_mock

        files = connector.fetch_code()
        assert len(files) == 0
