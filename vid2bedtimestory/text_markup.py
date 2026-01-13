from __future__ import annotations

import re


def to_reportlab_markup(text: str) -> str:
    """
    Convert lightweight emphasis markers into ReportLab Paragraph markup.

    Supported:
    - **bold**  -> <b>bold</b>
    - [bold]x[/bold] -> <b>x</b>
    """
    # Escape is intentionally minimal; ReportLab Paragraph supports a small XML subset.
    # We only insert <b> tags and leave other characters as-is.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # [bold]...[/bold]
    text = re.sub(r"\[bold\](.+?)\[/bold\]", r"<b>\1</b>", text, flags=re.IGNORECASE | re.DOTALL)

    # **...**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)

    # Convert newlines to breaks
    text = text.replace("\n", "<br/>")

    return text


