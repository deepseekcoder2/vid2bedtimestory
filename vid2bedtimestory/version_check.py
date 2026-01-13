from __future__ import annotations

import sys
from pathlib import Path

from .config import config


def check_system_requirements() -> None:
    """Check that all system requirements are met."""
    errors = []

    # Check Python version
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ required, found {sys.version}")

    # Check FFmpeg availability
    try:
        config.ffmpeg.get_ffprobe_path()
    except RuntimeError as e:
        errors.append(str(e))

    try:
        config.ffmpeg.get_ffmpeg_path()
    except RuntimeError as e:
        errors.append(str(e))

    if errors:
        error_msg = "System requirements not met:\n" + "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(error_msg)


def get_version_info() -> dict[str, str]:
    """Get version information for debugging."""
    from importlib.metadata import version as pkg_version

    def safe_version(pkg_name: str) -> str:
        """Safely get package version."""
        try:
            return pkg_version(pkg_name)
        except Exception:
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "reportlab": safe_version("reportlab"),
        "pillow": safe_version("pillow"),
        "srt": safe_version("srt"),
        "pydantic": safe_version("pydantic"),
        "typer": safe_version("typer"),
        "rich": safe_version("rich"),
        "ffprobe": str(config.ffmpeg.get_ffprobe_path()),
        "ffmpeg": str(config.ffmpeg.get_ffmpeg_path()),
    }
