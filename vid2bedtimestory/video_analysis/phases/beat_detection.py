"""
Phase 2: Beat Detection

LLM analyzes sparse captions + subtitles to identify story beats.
This is a text-only LLM call (no vision) that finds narrative structure.
"""

import json
from typing import Any, TYPE_CHECKING

from vid2bedtimestory.llm import call_with_json_mode, LLMError as BaseLLMError
from vid2bedtimestory.models import SubtitleSegment

from ..types import FrameCaption, BeatCandidate, LLMError
from ..prompts import BEAT_DETECTION_SYSTEM, BEAT_DETECTION_USER
from ..config import get_config
from ..subtitle_utils import format_subtitles_for_llm
from .sparse_survey import format_captions_for_llm

if TYPE_CHECKING:
    from vid2bedtimestory.knowledge import FranchiseData


def detect_beats(
    sparse_captions: list[FrameCaption],
    subtitles: list[SubtitleSegment],
    duration_s: float,
    franchise_db: "FranchiseData" = None,
) -> list[BeatCandidate]:
    """
    Identify 6-10 story beats from captions and dialogue.
    
    Uses an LLM to analyze the sparse frame captions and subtitle dialogue
    to identify the narrative structure of the video.
    
    Args:
        sparse_captions: List of FrameCaption from sparse survey phase
        subtitles: List of SubtitleSegment with dialogue
        duration_s: Total video duration in seconds
        franchise_db: Franchise database for beat examples
        
    Returns:
        List of BeatCandidate objects representing story segments
        
    Raises:
        LLMError: If LLM call fails or returns invalid output
    """
    config = get_config()
    
    # Step 1: Format inputs for LLM
    captions_text = format_captions_for_llm(sparse_captions)
    subtitles_text = format_subtitles_for_llm(subtitles, include_timestamps=True)
    
    # Calculate sample interval for prompt context
    sample_interval = duration_s / max(len(sparse_captions), 1)
    
    # Get franchise-specific beat examples
    franchise_beat_examples = ""
    if franchise_db:
        franchise_beat_examples = franchise_db.get_beat_examples()
    
    # Step 2: Build prompt
    user_prompt = BEAT_DETECTION_USER.format(
        duration_s=duration_s,
        sample_interval=sample_interval,
        captions_text=captions_text,
        subtitles_text=subtitles_text,
        franchise_beat_examples=franchise_beat_examples,
    )
    
    # Step 3: Call LLM with JSON mode
    try:
        response_data = call_with_json_mode(
            system_prompt=BEAT_DETECTION_SYSTEM,
            user_prompt=user_prompt,
            max_retries=3,
        )
    except BaseLLMError as e:
        raise LLMError(f"Beat detection LLM call failed: {e}") from e
    
    # Step 4: Parse response into BeatCandidate objects
    beats_data = response_data.get("beats", [])
    
    if not beats_data:
        raise LLMError("Beat detection returned no beats")
    
    beats: list[BeatCandidate] = []
    for i, beat_raw in enumerate(beats_data):
        try:
            beat = _parse_beat(beat_raw, i)
            beats.append(beat)
        except (KeyError, ValueError, TypeError) as e:
            raise LLMError(f"Failed to parse beat {i}: {e}") from e
    
    # Step 5: Sort beats by timestamp to ensure chronological processing
    beats.sort(key=lambda b: b.time_range[0])

    # Step 6: Validate beat coverage
    _validate_beats(beats, duration_s, config.min_beats, config.max_beats)
    
    return beats


def _parse_beat(beat_raw: dict[str, Any], index: int) -> BeatCandidate:
    """Parse a single beat from LLM response."""
    # Extract time_range (may be list or tuple)
    time_range = beat_raw.get("time_range", [0, 0])
    if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
        start_s = float(time_range[0])
        end_s = float(time_range[1])
    else:
        start_s = 0.0
        end_s = 0.0
    
    return BeatCandidate(
        beat_id=beat_raw.get("beat_id", f"beat_{index + 1:02d}"),
        beat_type=beat_raw.get("beat_type", "other"),
        summary=beat_raw.get("summary", ""),
        time_range=(start_s, end_s),
        anchor_dialogue=beat_raw.get("anchor_dialogue", []),
        confidence=float(beat_raw.get("confidence", 1.0)),
        missing_info_queries=beat_raw.get("missing_info_queries", []),
    )


def _validate_beats(
    beats: list[BeatCandidate],
    duration_s: float,
    min_beats: int,
    max_beats: int,
) -> None:
    """
    Validate that beats cover the timeline properly.
    
    Checks:
    - Number of beats within expected range
    - Beats cover from near start to near end
    - No major overlaps
    """
    if len(beats) < min_beats:
        raise LLMError(
            f"Too few beats detected: {len(beats)} < {min_beats}. "
            "Story structure may be unclear."
        )
    
    if len(beats) > max_beats:
        # Not a hard error, just truncate
        beats[:] = beats[:max_beats]
    
    # Check timeline coverage
    if beats:
        first_start = beats[0].time_range[0]
        last_end = beats[-1].time_range[1]
        
        # First beat should start within first 10% of video
        if first_start > duration_s * 0.1:
            # Not a hard error, but worth noting
            pass
        
        # Last beat should end within last 10% of video
        if last_end < duration_s * 0.9:
            # Not a hard error, but worth noting
            pass
    
    # Check for overlaps (beats should be sequential)
    for i in range(len(beats) - 1):
        current_end = beats[i].time_range[1]
        next_start = beats[i + 1].time_range[0]
        
        # Allow small overlap (1 second tolerance)
        if current_end > next_start + 1.0:
            # Overlapping beats - fix by adjusting boundary
            midpoint = (current_end + next_start) / 2
            beats[i] = BeatCandidate(
                beat_id=beats[i].beat_id,
                beat_type=beats[i].beat_type,
                summary=beats[i].summary,
                time_range=(beats[i].time_range[0], midpoint),
                anchor_dialogue=beats[i].anchor_dialogue,
                confidence=beats[i].confidence,
                missing_info_queries=beats[i].missing_info_queries,
            )
            beats[i + 1] = BeatCandidate(
                beat_id=beats[i + 1].beat_id,
                beat_type=beats[i + 1].beat_type,
                summary=beats[i + 1].summary,
                time_range=(midpoint, beats[i + 1].time_range[1]),
                anchor_dialogue=beats[i + 1].anchor_dialogue,
                confidence=beats[i + 1].confidence,
                missing_info_queries=beats[i + 1].missing_info_queries,
            )

