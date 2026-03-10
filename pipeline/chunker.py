"""Recursive text chunker — splits on headings, paragraphs, then sentences."""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int
    heading_context: str


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _split_on_headings(text: str) -> list[tuple[str, str]]:
    """Split text into (heading_context, section_body) pairs."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = stripped.lstrip("#").strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks respecting paragraph and sentence boundaries."""
    if _estimate_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try splitting on double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return _merge_splits(paragraphs, chunk_size, chunk_overlap)

    # Try splitting on single newlines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return _merge_splits(lines, chunk_size, chunk_overlap)

    # Fall back to sentence splitting
    sentences = []
    for part in text.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").splitlines():
        part = part.strip()
        if part:
            sentences.append(part)

    if len(sentences) > 1:
        return _merge_splits(sentences, chunk_size, chunk_overlap)

    # Last resort: hard split by character count
    char_limit = chunk_size * 4
    chunks = []
    for i in range(0, len(text), char_limit):
        chunks.append(text[i : i + char_limit])
    return chunks


def _merge_splits(parts: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Merge small parts into chunks up to chunk_size tokens, with overlap."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = _estimate_tokens(part)

        if current_tokens + part_tokens > chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts))

            # Keep trailing parts for overlap
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current_parts):
                p_tokens = _estimate_tokens(p)
                if overlap_tokens + p_tokens > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += p_tokens

            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(part)
        current_tokens += part_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[Chunk]:
    """Chunk text respecting document structure.

    Splits on headings first, then paragraphs, then sentences.
    Each chunk carries its heading context for retrieval relevance.
    """
    if not text.strip():
        return []

    sections = _split_on_headings(text)
    chunks: list[Chunk] = []
    index = 0

    for heading, body in sections:
        if not body.strip():
            continue

        sub_chunks = _split_text(body, chunk_size, chunk_overlap)
        for sub in sub_chunks:
            chunks.append(
                Chunk(
                    text=sub,
                    chunk_index=index,
                    heading_context=heading,
                )
            )
            index += 1

    return chunks
