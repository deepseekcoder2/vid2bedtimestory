"""
Phase 1: Sparse Survey

Sample uniformly-spaced frames and caption each with VLM.
This provides a quick overview of the video content for beat detection.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from ..types import FrameCaption, FrameExtractionError, VLMError
from ..prompts import get_sparse_survey_prompt, SPARSE_SURVEY_PROMPT
from ..config import get_config
from ..frame_utils import generate_uniform_timestamps, extract_frame_cached
from ..vlm_client import caption_frame, caption_frames_batch

if TYPE_CHECKING:
    from vid2bedtimestory.knowledge import FranchiseData


def sparse_survey(
    video_path: Path,
    duration_s: float,
    n_samples: int = None,
    franchise_db: "FranchiseData" = None,
) -> list[FrameCaption]:
    """
    Extract and caption uniformly-spaced frames.
    
    This is the first phase of video analysis. It samples frames at regular
    intervals and generates brief captions for each, providing a "sparse survey"
    of the video content.
    
    Args:
        video_path: Path to the video file
        duration_s: Total video duration in seconds
        n_samples: Number of frames to sample (defaults to config value)
        franchise_db: Franchise database for prompt injection
        
    Returns:
        List of FrameCaption objects with timestamps, paths, and captions
        
    Raises:
        FrameExtractionError: If frame extraction fails
        VLMError: If VLM captioning fails
    """
    config = get_config()
    n_samples = n_samples or config.sparse_survey_n_samples
    
    # Get franchise-specific prompt
    if franchise_db:
        prompt = get_sparse_survey_prompt(franchise_db)
    else:
        # Fallback to template (should not happen - franchise is required)
        prompt = SPARSE_SURVEY_PROMPT
    
    # Step 1: Generate uniform timestamps
    timestamps = generate_uniform_timestamps(
        duration_s=duration_s,
        n_samples=n_samples,
        start_offset_s=0.5,  # Skip potential black frames at start
        end_offset_s=0.5,    # Skip potential credits at end
    )
    
    if not timestamps:
        return []
    
    # Step 2: Extract frames at each timestamp
    frame_paths: list[Path] = []
    for ts in timestamps:
        try:
            frame_path = extract_frame_cached(video_path, ts)
            frame_paths.append(frame_path)
        except FrameExtractionError as e:
            # Re-raise with more context
            raise FrameExtractionError(
                f"Sparse survey failed: could not extract frame at {ts:.2f}s: {e}"
            ) from e
    
    # Step 3: Caption all frames using batch processing for efficiency
    # Build items list: (frame_path, prompt) tuples
    items = [(fp, prompt) for fp in frame_paths]
    
    try:
        captions = caption_frames_batch(items)
    except VLMError as e:
        raise VLMError(f"Sparse survey failed during VLM captioning: {e}") from e
    
    # Step 4: Build FrameCaption objects
    results: list[FrameCaption] = []
    for ts, frame_path, caption in zip(timestamps, frame_paths, captions):
        results.append(FrameCaption(
            timestamp_s=ts,
            frame_path=frame_path,
            caption=caption,
        ))
    
    return results


def format_captions_for_llm(captions: list[FrameCaption]) -> str:
    """
    Format sparse captions as text for LLM consumption.
    
    Args:
        captions: List of FrameCaption objects
        
    Returns:
        Formatted string with timestamp and caption per line
    """
    lines = []
    for fc in captions:
        lines.append(f"[{fc.timestamp_s:.1f}s] {fc.caption}")
    return "\n".join(lines)

