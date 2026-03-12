"""Deterministic content fingerprinting for incremental ingestion."""

import hashlib


def content_fingerprint(text: str) -> str:
    """Return a stable SHA-256 hex digest for the given text.

    Unlike Python's built-in hash(), this is deterministic across processes
    and interpreter restarts (not affected by PYTHONHASHSEED).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
