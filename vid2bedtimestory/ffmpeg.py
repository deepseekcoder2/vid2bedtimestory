from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import config


@dataclass(frozen=True)
class SubtitleStream:
    index: int
    codec_name: str | None
    language: str | None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # Use configured FFmpeg paths
    if cmd[0] == "ffprobe":
        cmd[0] = str(config.ffmpeg.get_ffprobe_path())
    elif cmd[0] == "ffmpeg":
        cmd[0] = str(config.ffmpeg.get_ffmpeg_path())

    return subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def ffprobe_streams(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        str(video_path),
    ]
    out = _run(cmd).stdout
    return json.loads(out)


def ffprobe_duration_s(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe.
    
    Tries video stream duration first (more accurate for MKV containers),
    then falls back to format/container duration.
    """
    # First try to get video stream duration from stream tags (more accurate for MKV)
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=DURATION",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        out = _run(cmd).stdout.strip()
        if out and out != "N/A" and ":" in out:
            # Parse duration in format HH:MM:SS.mmm (e.g., "00:10:18.618000000")
            parts = out.split(":")
            if len(parts) == 3:
                hours, mins, secs = parts
                stream_duration = float(hours) * 3600 + float(mins) * 60 + float(secs)
                if stream_duration > 0:
                    return stream_duration
    except (subprocess.CalledProcessError, ValueError):
        pass  # Fall through to format duration
    
    # Fall back to format/container duration
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ]
        out = _run(cmd).stdout
        data = json.loads(out)
        fmt = data.get("format") or {}
        duration = fmt.get("duration")
        if duration is None:
            raise RuntimeError(f"Could not read video duration from ffprobe for {video_path}")
        return float(duration)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed to analyze {video_path}: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON output from ffprobe for {video_path}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid duration value from ffprobe for {video_path}: {duration!r}") from e


def pick_subtitle_stream(video_path: Path, preferred_languages: list[str] | None = None) -> SubtitleStream:
    """Pick the best subtitle stream from the video."""
    preferred_languages = preferred_languages or ["eng", "en"]

    try:
        probe = ffprobe_streams(video_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to probe streams in {video_path}: {e.stderr}") from e

    streams = probe.get("streams", [])
    subtitle_streams: list[SubtitleStream] = []

    for s in streams:
        if s.get("codec_type") != "subtitle":
            continue
        try:
            tags = s.get("tags") or {}
            subtitle_streams.append(
                SubtitleStream(
                    index=int(s.get("index")),
                    codec_name=s.get("codec_name"),
                    language=(tags.get("language") or tags.get("LANGUAGE")),
                )
            )
        except (KeyError, ValueError) as e:
            # Skip malformed stream info
            continue

    if not subtitle_streams:
        available_streams = [s.get("codec_type", "unknown") for s in streams]
        raise RuntimeError(
            f"No subtitle streams found in {video_path}. "
            f"Available streams: {available_streams}. "
            "Video must have embedded subtitles."
        )

    # Prefer matching language first.
    for lang in preferred_languages:
        for ss in subtitle_streams:
            if (ss.language or "").lower() == lang.lower():
                return ss

    # Otherwise fall back to first subtitle stream.
    return subtitle_streams[0]


def extract_subtitles_to_srt(video_path: Path, out_srt_path: Path, stream_index: int) -> None:
    """Extract subtitles from video to SRT file."""
    try:
        out_srt_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-map",
            f"0:{stream_index}",
            # Force subtitle conversion to SRT
            "-c:s",
            "srt",
            str(out_srt_path),
        ]
        _run(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract subtitles from {video_path} stream {stream_index}: {e.stderr}"
        ) from e


def extract_frame(video_path: Path, timestamp_s: float, out_image_path: Path) -> None:
    """Extract a single frame from video at given timestamp."""
    try:
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        # -ss before -i is faster for most formats; accuracy is usually good enough for screenshots.
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp_s:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_image_path),
        ]
        _run(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract frame from {video_path} at {timestamp_s:.1f}s: {e.stderr}"
        ) from e


def extract_analysis_segment(
    video_path: Path,
    start_s: float,
    end_s: float,
    out_path: Path,
    target_height: int = 720,
    fps: float = 8.0,
    crf: int = 23,
) -> Path:
    """
    Extract a high-quality video segment for VLM analysis.
    
    Qwen3-VL-30B handles 720p video natively - extract at full quality
    so the VLM can properly analyze motion and details.
    
    Args:
        video_path: Source video
        start_s: Start time in seconds
        end_s: End time in seconds
        out_path: Output path for segment
        target_height: Height in pixels (720p native for Qwen3-VL)
        fps: Frames per second (8fps = smooth enough for action)
        crf: Quality (23 = high quality, visually lossless)
    
    Returns:
        Path to the extracted segment
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_s - start_s
    
    if duration <= 0:
        raise RuntimeError(f"Invalid segment duration: {start_s} to {end_s}")
    
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            # Scale down, reduce framerate
            "-vf", f"scale=-2:{target_height},fps={fps}",
            # Fast, low-quality encoding
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", str(crf),
            # No audio
            "-an",
            str(out_path),
        ]
        _run(cmd)
        
        if not out_path.exists():
            raise RuntimeError(f"Segment file not created: {out_path}")
            
        return out_path
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract segment [{start_s:.1f}s-{end_s:.1f}s] from {video_path}: {e.stderr}"
        ) from e


def extract_dense_frames(
    video_path: Path, 
    out_dir: Path, 
    interval_s: float = 0.5,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[Path]:
    """
    Extract frames from video at regular intervals.
    
    Creates frames named frame_NNNNNN.png where NNNNNN is the timestamp in milliseconds.
    This naming convention allows easy timestamp recovery from filenames.
    
    Args:
        video_path: Path to source video
        out_dir: Directory to save frames
        interval_s: Time between frames in seconds (default 0.5s = 2 fps)
        start_s: Start time in seconds
        end_s: End time in seconds (None = entire video)
    
    Returns:
        List of paths to extracted frames
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video duration if end_s not specified
    if end_s is None:
        end_s = ffprobe_duration_s(video_path)
    
    # Use fps filter for efficient extraction
    fps = 1.0 / interval_s
    
    try:
        # Build filter for time range and fps
        filters = [f"fps={fps:.4f}"]
        
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start_s:.3f}",
            "-i", str(video_path),
        ]
        
        # Add duration limit if end specified
        if end_s is not None and end_s > start_s:
            cmd.extend(["-t", f"{end_s - start_s:.3f}"])
        
        cmd.extend([
            "-vf", ",".join(filters),
            "-q:v", "2",
            # Output pattern: frame_000000.png, frame_000500.png, etc.
            # The %06d will be the frame number, we'll rename later
            str(out_dir / "temp_%06d.png"),
        ])
        
        _run(cmd)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract dense frames from {video_path}: {e.stderr}"
        ) from e
    
    # Rename frames to include timestamp in filename
    temp_frames = sorted(out_dir.glob("temp_*.png"))
    final_frames = []
    
    for i, temp_path in enumerate(temp_frames):
        timestamp_ms = int((start_s + i * interval_s) * 1000)
        final_name = f"frame_{timestamp_ms:06d}.png"
        final_path = out_dir / final_name
        temp_path.rename(final_path)
        final_frames.append(final_path)
    
    return final_frames
