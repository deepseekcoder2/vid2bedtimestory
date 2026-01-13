from __future__ import annotations

from dataclasses import dataclass

from .models import BookSpec, PageSpec, SubtitleSegment


@dataclass(frozen=True)
class FallbackOptions:
    target_pages: int = 22
    min_chars_per_page: int = 180
    max_chars_per_page: int = 420


def _chunk_segments_by_time(segments: list[SubtitleSegment], target_pages: int) -> list[list[SubtitleSegment]]:
    start = segments[0].start_ms
    end = segments[-1].end_ms
    total = max(1, end - start)
    bucket_ms = total / target_pages

    buckets: list[list[SubtitleSegment]] = [[] for _ in range(target_pages)]
    for seg in segments:
        mid = (seg.start_ms + seg.end_ms) / 2
        idx = int((mid - start) / bucket_ms)
        idx = max(0, min(target_pages - 1, idx))
        buckets[idx].append(seg)

    # Ensure no empty buckets by borrowing from neighbors.
    for i in range(target_pages):
        if buckets[i]:
            continue
        # Prefer pulling one segment from the nearest non-empty bucket.
        for d in range(1, target_pages):
            left = i - d
            right = i + d
            if left >= 0 and buckets[left]:
                buckets[i].append(buckets[left].pop())
                break
            if right < target_pages and buckets[right]:
                buckets[i].append(buckets[right].pop(0))
                break

    return buckets


def _segments_to_paragraphs(segs: list[SubtitleSegment], opts: FallbackOptions) -> list[str]:
    # Join subtitle lines into a single text block
    raw_text = " ".join(s.text for s in segs).strip()
    
    # 1. Truncate if massively too long (hard cap)
    if len(raw_text) > opts.max_chars_per_page:
        cut = raw_text[: opts.max_chars_per_page]
        raw_text = cut.strip() + "..."

    # 2. Semantic splitting heuristic
    # Split on distinct sentence endings to identify potential blocks
    # Then group them. New block if:
    # - Quote mark appears (Dialogue)
    # - Block gets too long (>2 sentences or >150 chars)
    
    sentences = raw_text.replace("!", "!<STOP>").replace("?", "?<STOP>").replace(".", ".<STOP>").split("<STOP>")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    paragraphs: list[str] = []
    current_para: list[str] = []
    
    for s in sentences:
        is_dialogue = '"' in s or "'" in s  # Rough check for quotes
        
        # If we have content, and (this is dialogue OR current para is long), break.
        if current_para and (is_dialogue or len(current_para) >= 3 or sum(len(x) for x in current_para) > 150):
            paragraphs.append(" ".join(current_para))
            current_para = []
            
        current_para.append(s)
        
    if current_para:
        paragraphs.append(" ".join(current_para))
        
    return paragraphs


def build_fallback_book(
    *,
    title: str,
    segments: list[SubtitleSegment],
    duration_s: float,
    opts: FallbackOptions,
) -> BookSpec:
    buckets = _chunk_segments_by_time(segments, opts.target_pages)
    pages: list[PageSpec] = []
    for i, segs in enumerate(buckets, start=1):
        paragraph_list = _segments_to_paragraphs(segs, opts)
        # Timestamp candidates: pick midpoint of bucket time range.
        start_ms = min(s.start_ms for s in segs)
        end_ms = max(s.end_ms for s in segs)
        mid_s = ((start_ms + end_ms) / 2) / 1000.0
        mid_s = max(0.0, min(duration_s - 0.5, mid_s))
        pages.append(
            PageSpec(
                page_index=i,
                paragraphs=paragraph_list,
                beat_type="other",
                image_timestamp_candidates_s=[mid_s],
                alt_text="",
                layout_hint="auto",
            )
        )
    return BookSpec(title=title, pages=pages)


