"""
Caching utilities for video analysis.

Provides hash-based caching for:
- Extracted video frames (PNG files)
- VLM caption results (JSON files)

Cache is stored in artifacts/cache/ by default (configurable).
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from .config import get_config


def _ensure_cache_dirs() -> None:
    """Create cache directories if they don't exist."""
    config = get_config()
    if not config.cache_enabled:
        return
    
    frames_dir = config.cache_dir / "frames"
    captions_dir = config.cache_dir / "captions"
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)


def video_hash(video_path: Path) -> str:
    """
    Generate a short hash for a video file.
    
    Uses filename + file size for speed (not content hash).
    This is sufficient since we're caching within a single project.
    
    Args:
        video_path: Path to video file
        
    Returns:
        12-character hex hash string
    """
    stat = video_path.stat()
    key = f"{video_path.name}:{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def prompt_hash(prompt: str) -> str:
    """
    Generate a short hash for a prompt string.
    
    Args:
        prompt: The prompt text
        
    Returns:
        8-character hex hash string
    """
    return hashlib.md5(prompt.encode()).hexdigest()[:8]


def frame_hash(frame_path: Path) -> str:
    """
    Generate a hash for an extracted frame.
    
    Uses file content hash for accuracy.
    
    Args:
        frame_path: Path to PNG frame file
        
    Returns:
        12-character hex hash string
    """
    content = frame_path.read_bytes()
    return hashlib.md5(content).hexdigest()[:12]


# =============================================================================
# FRAME CACHE
# =============================================================================

def get_frame_cache_path(video_path: Path, timestamp_s: float) -> Path:
    """
    Get the cache path for an extracted frame.
    
    Args:
        video_path: Source video file
        timestamp_s: Timestamp in seconds
        
    Returns:
        Path where frame PNG should be cached
    """
    config = get_config()
    vh = video_hash(video_path)
    return config.cache_dir / "frames" / f"{vh}_{timestamp_s:.2f}.png"


def get_cached_frame(video_path: Path, timestamp_s: float) -> Optional[Path]:
    """
    Check if a frame is cached and return its path.
    
    Args:
        video_path: Source video file
        timestamp_s: Timestamp in seconds
        
    Returns:
        Path to cached frame if exists, None otherwise
    """
    config = get_config()
    if not config.cache_enabled:
        return None
    
    cache_path = get_frame_cache_path(video_path, timestamp_s)
    if cache_path.exists():
        return cache_path
    return None


def cache_frame(video_path: Path, timestamp_s: float, frame_path: Path) -> Path:
    """
    Cache an extracted frame.
    
    Copies the frame to the cache directory.
    
    Args:
        video_path: Source video file
        timestamp_s: Timestamp in seconds
        frame_path: Path to the extracted frame (temp location)
        
    Returns:
        Path to the cached frame
    """
    config = get_config()
    if not config.cache_enabled:
        return frame_path
    
    _ensure_cache_dirs()
    cache_path = get_frame_cache_path(video_path, timestamp_s)
    
    # Copy if not already at cache path
    if frame_path != cache_path:
        import shutil
        shutil.copy2(frame_path, cache_path)
    
    return cache_path


# =============================================================================
# CAPTION CACHE
# =============================================================================

def get_caption_cache_path(frame_path: Path, prompt: str) -> Path:
    """
    Get the cache path for a caption result.
    
    Args:
        frame_path: Path to the frame that was captioned
        prompt: The prompt used for captioning
        
    Returns:
        Path where caption JSON should be cached
    """
    config = get_config()
    fh = frame_hash(frame_path)
    ph = prompt_hash(prompt)
    return config.cache_dir / "captions" / f"{fh}_{ph}.json"


def get_cached_caption(frame_path: Path, prompt: str) -> Optional[str]:
    """
    Check if a caption is cached and return it.
    
    Args:
        frame_path: Path to the frame
        prompt: The prompt used
        
    Returns:
        Cached caption text if exists, None otherwise
    """
    config = get_config()
    if not config.cache_enabled:
        return None
    
    cache_path = get_caption_cache_path(frame_path, prompt)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            return data.get("caption")
        except (json.JSONDecodeError, KeyError):
            # Invalid cache entry, will be regenerated
            return None
    return None


def cache_caption(frame_path: Path, prompt: str, caption: str) -> None:
    """
    Cache a caption result.
    
    Args:
        frame_path: Path to the frame
        prompt: The prompt used
        caption: The caption result to cache
    """
    config = get_config()
    if not config.cache_enabled:
        return
    
    _ensure_cache_dirs()
    cache_path = get_caption_cache_path(frame_path, prompt)
    
    data = {
        "frame_path": str(frame_path),
        "prompt_hash": prompt_hash(prompt),
        "caption": caption,
    }
    cache_path.write_text(json.dumps(data, indent=2))


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def clear_cache() -> int:
    """
    Clear all cached frames and captions.
    
    Returns:
        Number of files deleted
    """
    config = get_config()
    count = 0
    
    if config.cache_dir.exists():
        for cache_file in config.cache_dir.rglob("*"):
            if cache_file.is_file():
                cache_file.unlink()
                count += 1
    
    return count


def get_cache_stats() -> dict:
    """
    Get statistics about the cache.
    
    Returns:
        Dict with frame_count, caption_count, total_size_mb
    """
    config = get_config()
    
    frames_dir = config.cache_dir / "frames"
    captions_dir = config.cache_dir / "captions"
    
    frame_count = len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
    caption_count = len(list(captions_dir.glob("*.json"))) if captions_dir.exists() else 0
    
    total_size = 0
    if config.cache_dir.exists():
        for f in config.cache_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
    
    return {
        "frame_count": frame_count,
        "caption_count": caption_count,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }

