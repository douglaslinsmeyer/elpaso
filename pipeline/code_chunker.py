"""Language dispatcher for code chunking."""

from pipeline.csharp_chunker import CodeChunk, chunk_csharp


def chunk_code(source_code: str, language: str, chunk_size: int = 512) -> list[CodeChunk]:
    """Route to language-specific chunker.

    Currently supports C#. Other languages fall back to
    treating the whole file as a single chunk.
    """
    if language == "csharp":
        return chunk_csharp(source_code, chunk_size=chunk_size)

    # Fallback: whole file as one chunk
    text = source_code.strip()
    if not text:
        return []
    return [
        CodeChunk(
            text=text,
            chunk_index=0,
            namespace="",
            class_name="",
            method_name="",
            is_interface=False,
        )
    ]
