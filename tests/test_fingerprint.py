"""Tests for deterministic content fingerprinting."""

from pipeline.fingerprint import content_fingerprint


class TestContentFingerprint:
    def test_deterministic_across_calls(self):
        text = "public void ProcessShipment() {}"
        assert content_fingerprint(text) == content_fingerprint(text)

    def test_different_content_different_fingerprint(self):
        assert content_fingerprint("hello") != content_fingerprint("world")

    def test_unicode_handling(self):
        fp = content_fingerprint("café ñ 日本語")
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex digest length

    def test_empty_string(self):
        fp = content_fingerprint("")
        assert isinstance(fp, str)
        assert len(fp) == 64

    def test_returns_hex_string(self):
        fp = content_fingerprint("test")
        assert all(c in "0123456789abcdef" for c in fp)

    def test_whitespace_sensitivity(self):
        assert content_fingerprint("hello world") != content_fingerprint("hello  world")
