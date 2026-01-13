"""
Internal types for video analysis phases.

These types are used to pass data between phases. They are NOT part of
the public API - the final output is always AnalysisResult from models.py.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class VideoAnalysisError(Exception):
    """Base error for video analysis module."""
    pass


class FrameExtractionError(VideoAnalysisError):
    """Failed to extract frame from video."""
    pass


class VLMError(VideoAnalysisError):
    """VLM call failed or returned invalid output."""
    pass


class LLMError(VideoAnalysisError):
    """LLM call failed or returned invalid output."""
    pass


class ValidationError(VideoAnalysisError):
    """Output validation failed."""
    pass


class PhaseError(VideoAnalysisError):
    """A phase failed but may have partial results."""
    
    def __init__(self, message: str, phase: str, partial_result=None):
        super().__init__(message)
        self.phase = phase
        self.partial_result = partial_result


# =============================================================================
# PHASE 1: SPARSE SURVEY
# =============================================================================

class FrameCaption(BaseModel):
    """
    Phase 1 output: one frame, one caption.
    
    Represents a single uniformly-sampled frame with its VLM-generated caption.
    """
    timestamp_s: float = Field(..., description="Timestamp in seconds where frame was extracted")
    frame_path: Path = Field(..., description="Path to extracted PNG frame")
    caption: str = Field(..., description="VLM-generated description of the frame")
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# PHASE 2: BEAT DETECTION
# =============================================================================

class BeatCandidate(BaseModel):
    """
    Phase 2 output: a detected story segment.
    
    Represents a narrative beat identified by the LLM from sparse captions
    and subtitle dialogue.
    """
    beat_id: str = Field(..., description="Unique identifier (beat_01, beat_02, ...)")
    beat_type: str = Field(
        ..., 
        description="Story beat type: setup, inciting_incident, rising_action, climax, resolution"
    )
    summary: str = Field(..., description="1-2 sentence description of what happens")
    time_range: tuple[float, float] = Field(
        ..., 
        description="(start_s, end_s) - time window for this beat"
    )
    anchor_dialogue: list[str] = Field(
        default_factory=list,
        description="Key dialogue lines that anchor this beat"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for potential re-query (0-1)"
    )
    missing_info_queries: list[str] = Field(
        default_factory=list,
        description="Future: CLIP queries for additional frame retrieval"
    )


# =============================================================================
# PHASE 3: DEEP DIVE
# =============================================================================

class MomentCaption(BaseModel):
    """
    Phase 3 output: rich description of a key frame within a beat.
    
    Represents a densely-captioned frame with vivid visual description
    suitable for picture book illustration.
    """
    timestamp_s: float = Field(..., description="Timestamp in seconds")
    frame_path: Path = Field(..., description="Path to extracted PNG frame")
    visual_description: str = Field(
        ..., 
        min_length=50,
        description="50-150 word vivid description for illustrator"
    )
    emotional_beat: str = Field(..., description="Single emotion word: excitement, fear, triumph")
    key_dialogue: list[str] = Field(
        default_factory=list,
        description="Dialogue lines from nearby subtitles"
    )
    beat_id: str = Field(..., description="Links to parent BeatCandidate")
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# PHASE 4: CHARACTER EXTRACTION
# =============================================================================

class CharacterProfile(BaseModel):
    """
    Phase 4 output: detailed character information.
    
    Represents a character with explicit appearance description and pronouns,
    extracted via dedicated VLM pass on character reference frames.
    """
    name: str = Field(..., description="Character name as spoken in dialogue")
    role: str = Field(
        ...,
        description="Character role: protagonist, ally, mentor, antagonist, neutral"
    )
    traits: list[str] = Field(
        default_factory=list,
        description="2-4 personality traits visible from actions/expressions"
    )
    appearance: str = Field(
        ...,
        description="Physical description: gender, hair, clothing, age, features"
    )
    pronoun: str = Field(..., description="Pronoun: 'he', 'she', or 'they'")
    first_appearance_s: float = Field(..., description="Timestamp of first appearance")
    reference_frame_path: Path = Field(..., description="Frame used for appearance extraction")
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# ANALYSIS CONTEXT (State passed between phases)
# =============================================================================

class AnalysisContext(BaseModel):
    """
    State object passed between analysis phases.
    
    Accumulates results from each phase to be assembled into final output.
    """
    video_path: Path
    duration_s: float
    
    # Phase outputs (populated incrementally)
    sparse_captions: list[FrameCaption] = Field(default_factory=list)
    beats: list[BeatCandidate] = Field(default_factory=list)
    moments: list[MomentCaption] = Field(default_factory=list)
    characters: list[CharacterProfile] = Field(default_factory=list)
    
    # Metadata
    phase_completed: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

