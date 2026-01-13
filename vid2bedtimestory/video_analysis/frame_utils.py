"""
Frame extraction utilities for video analysis.

Provides high-level frame extraction with caching integration.
"""

from pathlib import Path
from typing import Optional
import tempfile

from ..ffmpeg import extract_frame as _ffmpeg_extract_frame
from .cache import get_cached_frame, cache_frame, get_frame_cache_path
from .config import get_config
from .types import FrameExtractionError


def extract_frame_cached(
    video_path: Path,
    timestamp_s: float,
    force: bool = False,
) -> Path:
    """
    Extract a frame from video, using cache if available.
    
    Args:
        video_path: Path to the video file
        timestamp_s: Timestamp in seconds (will be clamped to valid range)
        force: If True, bypass cache and re-extract
        
    Returns:
        Path to the extracted frame (PNG)
        
    Raises:
        FrameExtractionError: If FFmpeg fails to extract frame
    """
    # Clamp timestamp to reasonable bounds
    timestamp_s = max(0.0, timestamp_s)
    
    # Check cache first (unless forced)
    if not force:
        cached = get_cached_frame(video_path, timestamp_s)
        if cached:
            return cached
    
    # Extract to cache location directly
    config = get_config()
    
    if config.cache_enabled:
        out_path = get_frame_cache_path(video_path, timestamp_s)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Use temp file if caching disabled
        out_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".png").name)
    
    try:
        _ffmpeg_extract_frame(video_path, timestamp_s, out_path)
    except Exception as e:
        raise FrameExtractionError(
            f"Failed to extract frame at {timestamp_s:.2f}s from {video_path}: {e}"
        ) from e
    
    if not out_path.exists():
        raise FrameExtractionError(
            f"FFmpeg did not produce output file for {timestamp_s:.2f}s"
        )
    
    return out_path


def extract_frames_at_timestamps(
    video_path: Path,
    timestamps: list[float],
    force: bool = False,
) -> list[Path]:
    """
    Extract multiple frames at specified timestamps.
    
    Args:
        video_path: Path to the video file
        timestamps: List of timestamps in seconds
        force: If True, bypass cache and re-extract all
        
    Returns:
        List of paths to extracted frames (same order as timestamps)
        
    Raises:
        FrameExtractionError: If any frame fails to extract
    """
    return [
        extract_frame_cached(video_path, ts, force=force)
        for ts in timestamps
    ]


def generate_uniform_timestamps(
    duration_s: float,
    n_samples: int,
    start_offset_s: float = 0.5,
    end_offset_s: float = 0.5,
) -> list[float]:
    """
    Generate uniformly-spaced timestamps across a duration.
    
    Args:
        duration_s: Total duration in seconds
        n_samples: Number of samples to generate
        start_offset_s: Offset from start to avoid black frames
        end_offset_s: Offset from end to avoid credits
        
    Returns:
        List of timestamps in seconds
        
    Example:
        >>> generate_uniform_timestamps(600, 5)
        [0.5, 150.125, 299.75, 449.375, 599.0]
    """
    if n_samples < 1:
        return []
    
    if n_samples == 1:
        return [duration_s / 2]
    
    effective_start = start_offset_s
    effective_end = duration_s - end_offset_s
    effective_duration = effective_end - effective_start
    
    if effective_duration <= 0:
        # Duration too short, just return midpoint
        return [duration_s / 2]
    
    step = effective_duration / (n_samples - 1)
    return [effective_start + i * step for i in range(n_samples)]


def generate_timestamps_in_range(
    start_s: float,
    end_s: float,
    n_samples: int,
    anchor_timestamps: Optional[list[float]] = None,
) -> list[float]:
    """
    Generate timestamps within a specific time range.
    
    Optionally prioritizes anchor timestamps (e.g., dialogue moments).
    
    Args:
        start_s: Start of range in seconds
        end_s: End of range in seconds
        n_samples: Number of samples to generate
        anchor_timestamps: Priority timestamps to include (e.g., from subtitles)
        
    Returns:
        List of timestamps in seconds, sorted
    """
    if n_samples < 1:
        return []
    
    # Start with anchor timestamps that fall within range
    selected = []
    if anchor_timestamps:
        for ts in anchor_timestamps:
            if start_s <= ts <= end_s:
                selected.append(ts)
    
    # If we have enough anchors, sub-sample them uniformly to cover the full range
    if len(selected) >= n_samples:
        # Sort first to ensure uniform selection across the range
        selected.sort()
        # Pick n_samples evenly spaced indices
        indices = [int(i * (len(selected) - 1) / (n_samples - 1)) for i in range(n_samples)]
        return [selected[i] for i in indices]
    
    # Fill remaining with uniform samples
    remaining = n_samples - len(selected)
    duration = end_s - start_s
    
    if remaining > 0 and duration > 0:
        step = duration / (remaining + 1)
        for i in range(1, remaining + 1):
            candidate = start_s + i * step
            # Avoid duplicates with anchors (within 1 second tolerance)
            if not any(abs(candidate - a) < 1.0 for a in selected):
                selected.append(candidate)
    
    return sorted(selected[:n_samples])

