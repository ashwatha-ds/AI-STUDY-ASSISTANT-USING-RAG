from datetime import datetime


def format_sources(results: list) -> str:
    """Format source documents into a readable markdown string."""
    seen = set()
    lines = []

    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}-{page}"

        if key not in seen:
            seen.add(key)
            snippet = doc.page_content[:200].replace("\n", " ").strip()
            lines.append(f"**📄 {source}** — Page {page}\n> _{snippet}..._")

    return "\n\n".join(lines) if lines else ""


def export_chat(messages: list) -> str:
    """Export chat history as a plain text string."""
    lines = [
        "StudyMind AI — Chat Export",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        "",
    ]

    for msg in messages:
        role = "You" if msg["role"] == "user" else "StudyMind AI"
        lines.append(f"[{role}]")
        lines.append(msg["content"])
        if msg.get("mode"):
            src = "Document" if msg["mode"] == "doc" else "AI Knowledge"
            lines.append(f"(Source: {src})")
        lines.append("")

    return "\n".join(lines)
