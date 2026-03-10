"""Tests for the text chunker — boundary logic and structure preservation."""

from pipeline.chunker import Chunk, chunk_text, _estimate_tokens, _split_on_headings


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_text(self):
        # "hello" = 5 chars -> ~1 token
        assert _estimate_tokens("hello") == 1

    def test_longer_text(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


class TestSplitOnHeadings:
    def test_no_headings(self):
        sections = _split_on_headings("Just some plain text.\nMore text.")
        assert len(sections) == 1
        assert sections[0][0] == ""  # no heading
        assert "Just some plain text." in sections[0][1]

    def test_single_heading(self):
        text = "# Introduction\nSome content here."
        sections = _split_on_headings(text)
        assert len(sections) == 1
        assert sections[0][0] == "Introduction"
        assert sections[0][1] == "Some content here."

    def test_multiple_headings(self):
        text = "# First\nContent one.\n## Second\nContent two."
        sections = _split_on_headings(text)
        assert len(sections) == 2
        assert sections[0][0] == "First"
        assert sections[1][0] == "Second"

    def test_content_before_first_heading(self):
        text = "Preamble text.\n# Heading\nBody text."
        sections = _split_on_headings(text)
        assert len(sections) == 2
        assert sections[0][0] == ""
        assert "Preamble" in sections[0][1]


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n   ") == []

    def test_short_text_single_chunk(self):
        text = "A short paragraph."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_preserves_heading_context(self):
        text = "# Setup Guide\nInstall the software by running the install command."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].heading_context == "Setup Guide"

    def test_long_text_splits(self):
        # Create text that exceeds chunk_size
        paragraphs = [f"Paragraph {i}. " + ("word " * 100) for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=128, chunk_overlap=20)
        assert len(chunks) > 1
        # Verify sequential indexing
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_multiple_sections_chunk_independently(self):
        text = "# Section A\nContent for A.\n# Section B\nContent for B."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 2
        assert chunks[0].heading_context == "Section A"
        assert chunks[1].heading_context == "Section B"

    def test_chunk_indices_are_global(self):
        text = "# One\nText one.\n# Two\nText two.\n# Three\nText three."
        chunks = chunk_text(text, chunk_size=512)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_sections_skipped(self):
        text = "# Empty Section\n\n# Real Section\nActual content."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].heading_context == "Real Section"
