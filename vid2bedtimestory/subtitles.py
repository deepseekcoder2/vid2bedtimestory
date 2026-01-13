from __future__ import annotations

import re
from pathlib import Path

import srt

from .models import SubtitleSegment


_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    # Remove basic HTML-ish tags and normalize whitespace.
    text = _TAG_RE.sub("", text)
    text = text.replace("\u200e", "").replace("\u200f", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_srt_file(path: Path) -> list[SubtitleSegment]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    subs = list(srt.parse(raw))
    segments: list[SubtitleSegment] = []
    for item in subs:
        text = _clean_text(item.content)
        if not text:
            continue
        segments.append(
            SubtitleSegment(
                start_ms=int(item.start.total_seconds() * 1000),
                end_ms=int(item.end.total_seconds() * 1000),
                text=text,
            )
        )
    if not segments:
        raise RuntimeError("Extracted subtitles were empty after parsing/cleaning.")
    return segments


