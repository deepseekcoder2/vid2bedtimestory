from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FFmpegConfig(BaseModel):
    """Configuration for FFmpeg binary locations."""
    ffprobe_path: Optional[Path] = None
    ffmpeg_path: Optional[Path] = None

    def get_ffprobe_path(self) -> Path:
        """Get the path to ffprobe executable."""
        import platform
        
        if self.ffprobe_path:
            return self.ffprobe_path

        # Try to find in PATH (works on all platforms)
        ffprobe_in_path = shutil.which("ffprobe")
        if ffprobe_in_path:
            return Path(ffprobe_in_path)

        system = platform.system()
        
        if system == "Darwin":  # macOS
            # Try Homebrew locations
            homebrew_locations = [
                Path("/opt/homebrew/bin/ffprobe"),  # Apple Silicon Homebrew
                Path("/usr/local/bin/ffprobe"),      # Intel Homebrew
            ]
            for location in homebrew_locations:
                if location.exists():
                    return location
                    
            raise RuntimeError(
                "ffprobe not found. Please install FFmpeg via Homebrew:\n"
                "  brew install ffmpeg\n"
                "Or specify the path using --ffprobe-path or VID2BEDTIMESTORY_FFPROBE_PATH"
            )
        
        elif system == "Windows":
            # Try WinGet installation location specifically
            winget_base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
            if winget_base.exists():
                ffmpeg_packages = list(winget_base.glob("Gyan.FFmpeg*"))
                if ffmpeg_packages:
                    ffmpeg_package = sorted(ffmpeg_packages)[-1]
                    candidates = sorted(ffmpeg_package.glob("**/bin/ffprobe.exe"))
                    if candidates:
                        return candidates[-1]

            # Try common installation locations on Windows
            common_locations = [
                Path("C:") / "ffmpeg" / "bin",
                Path("C:") / "Program Files" / "FFmpeg" / "bin",
                Path("C:") / "Program Files (x86)" / "FFmpeg" / "bin",
            ]
            for location in common_locations:
                ffprobe_path = location / "ffprobe.exe"
                if ffprobe_path.exists():
                    return ffprobe_path

            raise RuntimeError(
                "ffprobe not found. Please ensure FFmpeg is installed and either:\n"
                "1. Add FFmpeg to your PATH\n"
                "2. Specify the path using --ffprobe-path\n"
                "3. Or use environment variables: VID2BEDTIMESTORY_FFPROBE_PATH=/path/to/ffprobe.exe"
            )
        
        else:  # Linux and others
            raise RuntimeError(
                "ffprobe not found. Please install FFmpeg:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  Fedora: sudo dnf install ffmpeg\n"
                "Or specify the path using --ffprobe-path or VID2BEDTIMESTORY_FFPROBE_PATH"
            )

    def get_ffmpeg_path(self) -> Path:
        """Get the path to ffmpeg executable."""
        import platform
        
        if self.ffmpeg_path:
            return self.ffmpeg_path

        # Try to find in PATH (works on all platforms)
        ffmpeg_in_path = shutil.which("ffmpeg")
        if ffmpeg_in_path:
            return Path(ffmpeg_in_path)

        system = platform.system()
        
        if system == "Darwin":  # macOS
            # Try Homebrew locations
            homebrew_locations = [
                Path("/opt/homebrew/bin/ffmpeg"),  # Apple Silicon Homebrew
                Path("/usr/local/bin/ffmpeg"),      # Intel Homebrew
            ]
            for location in homebrew_locations:
                if location.exists():
                    return location
                    
            raise RuntimeError(
                "ffmpeg not found. Please install FFmpeg via Homebrew:\n"
                "  brew install ffmpeg\n"
                "Or specify the path using --ffmpeg-path or VID2BEDTIMESTORY_FFMPEG_PATH"
            )
        
        elif system == "Windows":
            # Try WinGet installation location specifically
            winget_base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
            if winget_base.exists():
                ffmpeg_packages = list(winget_base.glob("Gyan.FFmpeg*"))
                if ffmpeg_packages:
                    ffmpeg_package = sorted(ffmpeg_packages)[-1]
                    candidates = sorted(ffmpeg_package.glob("**/bin/ffmpeg.exe"))
                    if candidates:
                        return candidates[-1]

            # Try common installation locations on Windows
            common_locations = [
                Path("C:") / "ffmpeg" / "bin",
                Path("C:") / "Program Files" / "FFmpeg" / "bin",
                Path("C:") / "Program Files (x86)" / "FFmpeg" / "bin",
            ]
            for location in common_locations:
                ffmpeg_path = location / "ffmpeg.exe"
                if ffmpeg_path.exists():
                    return ffmpeg_path

            raise RuntimeError(
                "ffmpeg not found. Please ensure FFmpeg is installed and either:\n"
                "1. Add FFmpeg to your PATH\n"
                "2. Specify the path using --ffmpeg-path\n"
                "3. Or use environment variables: VID2BEDTIMESTORY_FFMPEG_PATH=/path/to/ffmpeg.exe"
            )
        
        else:  # Linux and others
            raise RuntimeError(
                "ffmpeg not found. Please install FFmpeg:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  Fedora: sudo dnf install ffmpeg\n"
                "Or specify the path using --ffmpeg-path or VID2BEDTIMESTORY_FFMPEG_PATH"
            )


class LLMConfig(BaseModel):
    """Configuration for LLM API calls."""
    # Video analysis: Local MLX-VLM (Qwen3-VL-30B)
    # No API config needed - uses local model
    
    # Creative writing + pagination: Claude via OpenRouter
    creative_model: str = "anthropic/claude-sonnet-4.5"
    creative_base_url: str = "https://openrouter.ai/api/v1"
    
    # Utility tasks (visual query, simple extraction): cheap model
    utility_model: str = "deepseek/deepseek-v3.2"
    utility_model_fallback: str = "google/gemini-2.5-flash"
    
    # Frame scoring VLM: Can be "local" (MLX-VLM) or "cloud" (OpenRouter)
    vlm_backend: str = "cloud"  # "local" or "cloud"
    vlm_cloud_model: str = "qwen/qwen3-vl-235b-a22b-instruct"  # OpenRouter vision model
    vlm_cloud_max_concurrent: int = 60  # Parallel API requests for cloud scoring
    vlm_min_score_threshold: float = 5.0  # Minimum VLM score for frame selection
    
    max_retries: int = 3
    timeout: int = 600  # 10 minutes for video analysis
    temperature_story: float = 0.3  # Moderate creativity for story writing
    temperature_pagination: float = 0.1  # Low creativity for pagination


class AppConfig(BaseModel):
    """Main configuration for vid2bedtimestory."""
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Default values matching PRD
    default_pages_target: int = 22
    default_pages_min: int = 18
    # Development default: allow larger books (pagination may over-split while prompts evolve)
    default_pages_max: int = 40
    default_age_range: str = "5-8"
    default_lang: str = "eng"

    # Cache and artifacts
    cache_dir: Path = Field(default_factory=lambda: Path("cache"))
    artifacts_dir_default: str = "artifacts"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables."""
        config = cls()

        # Allow overriding FFmpeg paths via environment
        if ffprobe_path := os.getenv("VID2BEDTIMESTORY_FFPROBE_PATH"):
            config.ffmpeg.ffprobe_path = Path(ffprobe_path)
        if ffmpeg_path := os.getenv("VID2BEDTIMESTORY_FFMPEG_PATH"):
            config.ffmpeg.ffmpeg_path = Path(ffmpeg_path)

        # Allow overriding defaults via environment
        if cache_dir := os.getenv("VID2BEDTIMESTORY_CACHE_DIR"):
            config.cache_dir = Path(cache_dir)

        return config


# Global config instance
config = AppConfig.from_env()
