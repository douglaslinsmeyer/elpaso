"""System and synthesis prompt templates for El Paso Q&A."""

SYSTEM_PROMPT = """\
You are El Paso, an expert on the Ping Golf manufacturing software systems.
You have deep knowledge of the C#/.NET microservices architecture, RabbitMQ messaging, \
PostgreSQL databases, Blazor frontends, and all related processes documented in Confluence.

Rules:
- Answer ONLY from the provided context chunks. Never use outside knowledge.
- For every claim, cite the source using [Source N] notation.
- If the context is insufficient to fully answer, say so explicitly.
- Be concise and direct. Developers are your audience.
- When referencing code, include the namespace/class/method path.
"""


def build_synthesis_prompt(question: str, chunks: list[dict]) -> str:
    """Build the user prompt with retrieved context chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_type = chunk.get("source_type", "unknown")
        title = chunk.get("page_title") or chunk.get("title") or chunk.get("repo_name", "")
        file_path = chunk.get("file_path", "")
        class_name = chunk.get("class_name", "")
        method_name = chunk.get("method_name", "")
        heading = chunk.get("heading_context", "")
        url = chunk.get("page_url") or chunk.get("repo_url", "")
        text = chunk.get("text", "")

        # Build source label
        label_parts = [f"[{source_type}]"]
        if title:
            label_parts.append(title)
        if file_path:
            label_parts.append(file_path)
        if class_name:
            label_parts.append(f"Class: {class_name}")
        if method_name:
            label_parts.append(f"Method: {method_name}")
        if heading:
            label_parts.append(f"Section: {heading}")

        source_label = " | ".join(label_parts)

        context_parts.append(
            f"[Source {i}] {source_label}\nURL: {url}\n{text}"
        )

    context = "\n---\n".join(context_parts)

    return f"""\
Context:
---
{context}
---

Question: {question}

Provide a clear, concise answer citing sources with [Source N] notation."""
