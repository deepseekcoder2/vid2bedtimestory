"""
Phase 3: Deep Dive

Dense captioning of 3-5 frames per beat.
Gets rich visual descriptions for each beat's key moments.
"""

import json
from pathlib import Path
from typing import Optional

from vid2bedtimestory.models import SubtitleSegment
from vid2bedtimestory.knowledge import FranchiseData

from ..types import BeatCandidate, MomentCaption, FrameExtractionError, VLMError
from ..prompts import DEEP_DIVE_PROMPT
from ..config import get_config
from ..frame_utils import extract_frame_cached, generate_timestamps_in_range
from ..subtitle_utils import get_dialogue_timestamps, get_dialogue_texts_in_range
from ..vlm_client import caption_frames_batch, extract_json_robust


def deep_dive(
    video_path: Path,
    beats: list[BeatCandidate],
    subtitles: list[SubtitleSegment],
    frames_per_beat: int = None,
    franchise_db: Optional[FranchiseData] = None,
) -> list[MomentCaption]:
    """
    Get rich visual descriptions for each beat's key moments.
    
    For each story beat, this phase:
    1. Identifies key timestamps (anchored to dialogue when possible)
    2. Extracts frames at those timestamps
    3. Generates vivid visual descriptions suitable for illustration
    
    Args:
        video_path: Path to the video file
        beats: List of BeatCandidate from beat detection phase
        subtitles: List of SubtitleSegment with dialogue
        frames_per_beat: Number of frames per beat (defaults to config value)
        franchise_db: Optional franchise database for character context
        
    Returns:
        List of MomentCaption objects with rich visual descriptions
        
    Raises:
        FrameExtractionError: If frame extraction fails
        VLMError: If VLM captioning fails
    """
    config = get_config()
    
    # Step 1: Sort beats by timestamp to ensure we process them in order
    sorted_beats = sorted(beats, key=lambda b: b.time_range[0])
    
    # Step 2: Calculate dynamic frames_per_beat based on budget
    # We want to distribute config.max_moments across all beats fairly.
    if not sorted_beats:
        return []
        
    avg_per_beat = config.max_moments // len(sorted_beats)
    remainder = config.max_moments % len(sorted_beats)
    
    # Clamp to reasonable bounds
    avg_per_beat = max(2, min(8, avg_per_beat))
    
    all_moments: list[MomentCaption] = []
    
    # Process each beat
    for i, beat in enumerate(sorted_beats):
        # Distribute remainder to the first few beats
        n_samples = avg_per_beat + (1 if i < remainder else 0)
        
        beat_moments = _process_beat(
            video_path=video_path,
            beat=beat,
            subtitles=subtitles,
            frames_per_beat=n_samples,
            franchise_db=franchise_db,
        )
        all_moments.extend(beat_moments)
        
        # We no longer break early! We ensure every beat gets its share.
        # But we still respect a hard cap for safety if things go weird.
        if len(all_moments) >= config.max_moments + 10:
            break
    
    # Ensure we have minimum moments
    if len(all_moments) < config.min_moments:
        # Could add additional sampling here, but for now just return what we have
        pass
    
    return all_moments


def _process_beat(
    video_path: Path,
    beat: BeatCandidate,
    subtitles: list[SubtitleSegment],
    frames_per_beat: int,
    franchise_db: Optional[FranchiseData] = None,
) -> list[MomentCaption]:
    """
    Process a single beat: extract frames and caption them.
    """
    start_s, end_s = beat.time_range
    
    # Step 1: Get dialogue timestamps within this beat (for anchoring)
    dialogue_timestamps = get_dialogue_timestamps(
        subtitles, start_s, end_s, exclude_sound_effects=True
    )
    
    # Step 2: Generate frame timestamps, prioritizing dialogue moments
    timestamps = generate_timestamps_in_range(
        start_s=start_s,
        end_s=end_s,
        n_samples=frames_per_beat,
        anchor_timestamps=dialogue_timestamps,
    )
    
    if not timestamps:
        return []
    
    # Step 3: Extract frames
    frame_paths: list[Path] = []
    valid_timestamps: list[float] = []
    
    for ts in timestamps:
        try:
            frame_path = extract_frame_cached(video_path, ts)
            frame_paths.append(frame_path)
            valid_timestamps.append(ts)
        except FrameExtractionError:
            # Skip frames that fail to extract
            continue
    
    if not frame_paths:
        return []
    
    # Step 4: Build prompts with dialogue context and franchise reference
    items: list[tuple[Path, str]] = []
    dialogue_contexts: list[str] = []
    
    # Format character reference if database is available
    char_ref = ""
    franchise_deep_dive_examples = ""
    if franchise_db:
        char_lines = []
        for cid, info in franchise_db.characters.items():
            vis = info.visual_signature
            char_lines.append(f"- {info.display_name} ({info.role}): {vis.get('hair', '')}, {vis.get('clothing', '')}, {vis.get('distinctive', '')}")
        char_ref = "REFERENCE DATA: Use these character details for accuracy:\n" + "\n".join(char_lines)
        
        # Get franchise-specific deep dive examples
        franchise_deep_dive_examples = franchise_db.get_deep_dive_examples()

    for ts, frame_path in zip(valid_timestamps, frame_paths):
        # Get dialogue near this timestamp (±5 seconds)
        nearby_dialogue = get_dialogue_texts_in_range(
            subtitles, ts - 5.0, ts + 5.0, exclude_sound_effects=True
        )
        dialogue_context = "\n".join(nearby_dialogue) if nearby_dialogue else "(no dialogue)"
        dialogue_contexts.append(dialogue_context)
        
        # Format prompt with context, reference, and franchise examples
        prompt = DEEP_DIVE_PROMPT.format(
            dialogue_context=dialogue_context,
            character_reference=char_ref,
            franchise_deep_dive_examples=franchise_deep_dive_examples,
        )
        items.append((frame_path, prompt))
    
    # Step 5: Batch caption all frames
    # Force cloud=True for deep dive to ensure high quality
    try:
        captions = caption_frames_batch(items, prefer_cloud=True)
    except VLMError as e:
        raise VLMError(f"Deep dive captioning failed for beat {beat.beat_id}: {e}") from e
    
    # Step 6: Parse responses and build MomentCaption objects
    moments: list[MomentCaption] = []
    
    for ts, frame_path, caption_raw, dialogue_context in zip(
        valid_timestamps, frame_paths, captions, dialogue_contexts
    ):
        parsed = extract_json_robust(caption_raw)
        
        # Get visual description, falling back to raw caption
        visual_desc = parsed.get("visual_description", "") or caption_raw or ""
        
        # Skip frames with empty or too-short descriptions (VLM failure)
        if len(visual_desc.strip()) < 50:
            print(f"[deep_dive] WARNING: Skipping frame at {ts:.1f}s - VLM returned insufficient description ({len(visual_desc)} chars)")
            continue
        
        # Get key dialogue for this moment
        key_dialogue = get_dialogue_texts_in_range(
            subtitles, ts - 2.0, ts + 2.0, exclude_sound_effects=True
        )
        
        moments.append(MomentCaption(
            timestamp_s=ts,
            frame_path=frame_path,
            visual_description=visual_desc,
            emotional_beat=parsed.get("emotional_beat", "neutral"),
            key_dialogue=key_dialogue[:3],  # Limit to 3 lines
            beat_id=beat.beat_id,
        ))
    
    return moments

