"""
Refinement module for video analysis.

Post-assembly check for gaps. Can request additional frames if coverage is poor.
"""

from pathlib import Path
from typing import Optional

from vid2bedtimestory.models import AnalysisResult, SubtitleSegment, Moment

from .types import MomentCaption, FrameExtractionError, VLMError
from .prompts import DEEP_DIVE_PROMPT
from .frame_utils import extract_frame_cached
from .subtitle_utils import get_subtitles_in_range, get_dialogue_texts_in_range
from .vlm_client import caption_frame, extract_json_robust

import json


def check_coverage(
    analysis: AnalysisResult,
    subtitles: list[SubtitleSegment],
) -> dict:
    """
    Check how well moments cover the dialogue.
    
    Args:
        analysis: The assembled AnalysisResult
        subtitles: List of subtitle segments
        
    Returns:
        Dictionary with:
        - dialogue_coverage: fraction of dialogue lines captured (0-1)
        - uncovered_segments: list of (timestamp, text) for missed dialogue
        - time_gaps: list of (start, end) tuples for gaps >30s with no moments
    """
    moments = analysis.moments
    
    # Build set of covered time ranges (±3 seconds around each moment)
    covered_ranges = []
    for m in moments:
        ts = m.timestamp_range[0]
        covered_ranges.append((ts - 3, ts + 3))
    
    # Check each subtitle for coverage
    covered_count = 0
    uncovered_segments = []
    
    for sub in subtitles:
        ts = sub.start_ms / 1000.0
        text = sub.text.strip()
        
        # Skip sound effects
        if text.startswith("[") and text.endswith("]"):
            continue
        
        # Check if covered by any moment
        is_covered = any(
            start <= ts <= end 
            for start, end in covered_ranges
        )
        
        if is_covered:
            covered_count += 1
        else:
            uncovered_segments.append((ts, text))
    
    # Calculate dialogue coverage
    total_dialogue = len([
        s for s in subtitles 
        if not (s.text.strip().startswith("[") and s.text.strip().endswith("]"))
    ])
    dialogue_coverage = covered_count / max(total_dialogue, 1)
    
    # Find time gaps (>30s between moments)
    time_gaps = []
    if moments:
        sorted_moments = sorted(moments, key=lambda m: m.timestamp_range[0])
        
        for i in range(len(sorted_moments) - 1):
            current_end = sorted_moments[i].timestamp_range[1]
            next_start = sorted_moments[i + 1].timestamp_range[0]
            
            gap = next_start - current_end
            if gap > 30:
                time_gaps.append((current_end, next_start))
    
    return {
        "dialogue_coverage": dialogue_coverage,
        "uncovered_segments": uncovered_segments[:20],  # Limit to 20
        "time_gaps": time_gaps,
    }


def refine_if_needed(
    analysis: AnalysisResult,
    subtitles: list[SubtitleSegment],
    video_path: Path,
    min_coverage: float = 0.5,
) -> AnalysisResult:
    """
    If coverage < min_coverage, extract additional frames at uncovered timestamps.
    
    Args:
        analysis: The assembled AnalysisResult
        subtitles: List of subtitle segments
        video_path: Path to video file for additional frame extraction
        min_coverage: Minimum dialogue coverage threshold
        
    Returns:
        Enriched AnalysisResult with additional moments if needed
    """
    # Step 1: Check coverage
    coverage_info = check_coverage(analysis, subtitles)
    
    # Step 2: If adequate, return unchanged
    if coverage_info["dialogue_coverage"] >= min_coverage:
        return analysis
    
    print(f"[refinement] Coverage {coverage_info['dialogue_coverage']:.1%} < {min_coverage:.1%}, adding moments...")
    
    # Step 3: Identify candidate timestamps for refinement
    # Priority 1: Uncovered dialogue
    # Priority 2: Large time gaps (>30s) even if silent (e.g., action scenes)
    
    candidates: list[tuple[float, str]] = []
    
    # Add uncovered dialogue
    for ts, text in coverage_info["uncovered_segments"]:
        candidates.append((ts, f"dialogue: {text}"))
        
    # Add midpoints of large time gaps
    for start, end in coverage_info["time_gaps"]:
        midpoint = (start + end) / 2
        candidates.append((midpoint, "silent gap"))
        
    # Sort candidates by timestamp to process chronologically
    candidates.sort()
    
    # Limit additional frames to prevent runaway
    max_additional = 10
    additional_moments = []
    
    # Track timestamps we've already added to avoid duplicates
    added_timestamps: list[float] = []
    
    for ts, reason in candidates:
        if len(additional_moments) >= max_additional:
            break
            
        # Avoid adding if too close to an existing or already added moment
        if any(abs(ts - existing_ts) < 10.0 for existing_ts in added_timestamps):
            continue
            
        try:
            moment = _extract_additional_moment(
                video_path=video_path,
                timestamp_s=ts,
                subtitles=subtitles,
                moment_index=len(analysis.moments) + len(additional_moments),
            )
            if moment:
                additional_moments.append(moment)
                added_timestamps.append(ts)
                print(f"[refinement] Added moment at {ts:.1f}s (Reason: {reason})")
        except (FrameExtractionError, VLMError) as e:
            print(f"[refinement] Failed to extract moment at {ts:.1f}s: {e}")
            continue
    
    # Step 4: Return enriched analysis
    if additional_moments:
        # Combine and sort moments
        all_moments = list(analysis.moments) + additional_moments
        all_moments.sort(key=lambda m: m.timestamp_range[0])
        
        return AnalysisResult(
            title_candidates=analysis.title_candidates,
            characters=analysis.characters,
            beats=analysis.beats,
            moments=all_moments,
        )
    
    return analysis


def _extract_additional_moment(
    video_path: Path,
    timestamp_s: float,
    subtitles: list[SubtitleSegment],
    moment_index: int,
) -> Optional[Moment]:
    """
    Extract a single additional moment at a given timestamp.
    """
    # Extract frame
    frame_path = extract_frame_cached(video_path, timestamp_s)
    
    # Get dialogue context
    nearby_dialogue = get_dialogue_texts_in_range(
        subtitles, timestamp_s - 5, timestamp_s + 5, exclude_sound_effects=True
    )
    dialogue_context = "\n".join(nearby_dialogue) if nearby_dialogue else "(no dialogue)"
    
    # Caption frame
    prompt = DEEP_DIVE_PROMPT.format(
        dialogue_context=dialogue_context,
        character_reference="" # Refinement doesn't use reference for simplicity
    )
    # Force cloud=True for quality refinement
    response = caption_frame(frame_path, prompt, prefer_cloud=True)
    
    # Parse response
    parsed = extract_json_robust(response)
    
    # Get key dialogue
    key_dialogue = get_dialogue_texts_in_range(
        subtitles, timestamp_s - 2, timestamp_s + 2, exclude_sound_effects=True
    )
    
    return Moment(
        moment_id=f"moment_{moment_index + 1:03d}",
        beat_type="other",  # Refined moments don't have clear beat association
        timestamp_range=(timestamp_s, timestamp_s + 5),
        visual_description=parsed.get("visual_description", response),
        key_dialogue=key_dialogue[:3],
        screenshot_candidates_s=[
            max(0, timestamp_s - 1),
            timestamp_s,
            timestamp_s + 1,
        ],
        emotional_beat=parsed.get("emotional_beat", "neutral"),
    )

