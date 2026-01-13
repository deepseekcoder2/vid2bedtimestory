"""
Configuration for video analysis module.

All tunable parameters are centralized here for easy adjustment.
"""

from pathlib import Path
from pydantic import BaseModel, Field


class VideoAnalysisConfig(BaseModel):
    """Configuration for video analysis pipeline."""
    
    # ==========================================================================
    # PHASE 1: SPARSE SURVEY
    # ==========================================================================
    sparse_survey_n_samples: int = Field(
        default=28,
        ge=5,
        le=36,
        description="Number of uniformly-sampled frames for initial survey. "
                    "Higher values improve beat detection at marginal cost (~1 VLM call per 10 frames)"
    )
    
    # ==========================================================================
    # PHASE 2: BEAT DETECTION
    # ==========================================================================
    min_beats: int = Field(
        default=5,
        ge=3,
        description="Minimum number of story beats to detect"
    )
    max_beats: int = Field(
        default=12,
        le=20,
        description="Maximum number of story beats to detect"
    )
    
    # ==========================================================================
    # PHASE 3: DEEP DIVE
    # ==========================================================================
    frames_per_beat: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of frames to caption per story beat"
    )
    min_moments: int = Field(
        default=20,
        ge=10,
        description="Minimum total moments across all beats"
    )
    max_moments: int = Field(
        default=50,
        le=100,
        description="Maximum total moments to prevent runaway generation"
    )
    
    # ==========================================================================
    # PHASE 4: CHARACTER EXTRACTION
    # ==========================================================================
    max_characters: int = Field(
        default=10,
        le=20,
        description="Maximum characters to extract"
    )
    
    # ==========================================================================
    # VLM SETTINGS
    # ==========================================================================
    vlm_timeout_s: int = Field(
        default=120,
        ge=30,
        description="Timeout for single VLM call in seconds"
    )
    vlm_max_tokens: int = Field(
        default=1024,
        ge=128,
        le=2048,
        description="Maximum tokens for VLM response"
    )
    vlm_batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of frames to process in single VLM subprocess"
    )
    
    # ==========================================================================
    # CACHING
    # ==========================================================================
    cache_enabled: bool = Field(
        default=True,
        description="Whether to cache frames and captions"
    )
    cache_dir: Path = Field(
        default=Path("artifacts/cache"),
        description="Directory for cached frames and captions"
    )
    
    # ==========================================================================
    # VALIDATION THRESHOLDS
    # ==========================================================================
    min_dialogue_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of dialogue lines to capture"
    )
    max_duplicate_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Maximum fraction of duplicate descriptions allowed"
    )
    max_time_gap_s: float = Field(
        default=60.0,
        ge=10.0,
        description="Maximum allowed gap between moments in seconds"
    )
    
    class Config:
        arbitrary_types_allowed = True


# Global config instance (can be overridden for testing)
config = VideoAnalysisConfig()


def get_config() -> VideoAnalysisConfig:
    """Get the current configuration."""
    return config


def set_config(new_config: VideoAnalysisConfig) -> None:
    """Set a new configuration (useful for testing)."""
    global config
    config = new_config

