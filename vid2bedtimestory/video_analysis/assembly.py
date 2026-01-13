"""
Assembly module for video analysis.

Combines all phase outputs into the final AnalysisResult.
"""

import json
from pathlib import Path
from typing import Optional

from vid2bedtimestory.llm import call_with_json_mode, LLMError as BaseLLMError
from vid2bedtimestory.models import AnalysisResult, Character, Beat, Moment, BeatType

from .types import (
    FrameCaption, BeatCandidate, MomentCaption, CharacterProfile,
    LLMError, ValidationError
)
from .prompts import TITLE_GENERATION_PROMPT
from .validation import validate_all


# Beat type normalization (mirrors vid2bedtimestory/llm.py)
VALID_BEAT_TYPES = {"setup", "inciting_incident", "rising_action", "climax", "resolution", "other"}
BEAT_TYPE_ALIASES = {
    "crisis": "climax",
    "falling_action": "resolution",
    "exposition": "setup",
    "conflict": "rising_action",
    "denouement": "resolution",
    "inciting incident": "inciting_incident",
    "inciting-incident": "inciting_incident",
    "rising action": "rising_action",
    "rising-action": "rising_action",
    "falling action": "resolution",
}


def normalize_beat_type(beat_type: str) -> BeatType:
    """
    Normalize a beat type string to a valid BeatType literal.
    
    Args:
        beat_type: Raw beat type string from LLM
        
    Returns:
        Valid BeatType literal
    """
    bt = beat_type.lower().strip().replace("-", "_").replace(" ", "_")
    bt = BEAT_TYPE_ALIASES.get(bt, bt)
    
    if bt in VALID_BEAT_TYPES:
        return bt  # type: ignore
    return "other"


def generate_title_candidates(
    beats: list[BeatCandidate],
    protagonist: str,
) -> list[str]:
    """
    Generate 3-5 title options using LLM.
    
    Args:
        beats: List of story beats for context
        protagonist: Name of main character
        
    Returns:
        List of title candidate strings
    """
    # Build beats summary
    beats_summary = "\n".join(
        f"- {b.beat_type}: {b.summary}" for b in beats
    )
    
    prompt = TITLE_GENERATION_PROMPT.format(
        beats_summary=beats_summary,
        protagonist_name=protagonist or "the main character",
    )
    
    try:
        response = call_with_json_mode(
            system_prompt="You are a children's book title generator.",
            user_prompt=prompt,
            max_retries=2,
        )
        
        titles = response.get("titles", [])
        if isinstance(titles, list):
            return [str(t) for t in titles[:5]]
        return []
        
    except BaseLLMError:
        # Fallback: generate generic title
        return [
            f"{protagonist}'s Adventure" if protagonist else "An Amazing Adventure",
            "A Story of Courage",
            "The Big Day",
        ]


def assemble_analysis(
    sparse_captions: list[FrameCaption],
    beats: list[BeatCandidate],
    moments: list[MomentCaption],
    characters: list[CharacterProfile],
    duration_s: float,
) -> AnalysisResult:
    """
    Combine all phase outputs into AnalysisResult.
    
    Args:
        sparse_captions: Phase 1 output (unused in final result, but useful for validation)
        beats: Phase 2 output - story beats
        moments: Phase 3 output - moment captions
        characters: Phase 4 output - character profiles
        duration_s: Total video duration
        
    Returns:
        AnalysisResult ready for downstream consumers
        
    Raises:
        ValidationError: If validation fails
        LLMError: If title generation fails
    """
    # Step 1: Validate all inputs
    warnings = validate_all(beats, moments, characters, duration_s)
    
    # Log warnings (could be captured by caller)
    for warning in warnings:
        print(f"[assembly] Warning: {warning}")
    
    # Step 2: Find protagonist for title generation
    protagonist = _find_protagonist(characters)
    
    # Step 3: Generate title candidates
    title_candidates = generate_title_candidates(beats, protagonist)
    
    # Step 4: Convert BeatCandidate → Beat
    converted_beats = _convert_beats(beats)
    
    # Step 5: Convert MomentCaption → Moment
    converted_moments = _convert_moments(moments, beats)
    
    # Step 6: Convert CharacterProfile → Character
    converted_characters = _convert_characters(characters)
    
    # Step 7: Build and return AnalysisResult
    return AnalysisResult(
        title_candidates=title_candidates,
        characters=converted_characters,
        beats=converted_beats,
        moments=converted_moments,
    )


def _find_protagonist(characters: list[CharacterProfile]) -> str:
    """Find the protagonist character name."""
    for char in characters:
        if char.role == "protagonist":
            return char.name
    
    # Fallback to first character
    if characters:
        return characters[0].name
    
    return ""


def _convert_beats(beats: list[BeatCandidate]) -> list[Beat]:
    """Convert BeatCandidate objects to Beat objects."""
    return [
        Beat(
            beat_type=normalize_beat_type(bc.beat_type),
            summary=bc.summary,
            timestamp_range=bc.time_range,
        )
        for bc in beats
    ]


def _convert_moments(
    moments: list[MomentCaption],
    beats: list[BeatCandidate],
) -> list[Moment]:
    """
    Convert MomentCaption objects to Moment objects.
    
    Adds moment_id and screenshot_candidates_s.
    """
    # Build beat lookup for beat_type
    beat_lookup = {b.beat_id: b for b in beats}
    
    converted = []
    for i, mc in enumerate(moments):
        # Get beat type from parent beat
        parent_beat = beat_lookup.get(mc.beat_id)
        beat_type = normalize_beat_type(parent_beat.beat_type) if parent_beat else "other"
        
        # Generate screenshot candidates around the moment timestamp
        ts = mc.timestamp_s
        screenshot_candidates = [
            max(0, ts - 1),  # 1 second before
            ts,              # exact timestamp
            ts + 1,          # 1 second after
        ]
        
        converted.append(Moment(
            moment_id=f"moment_{i+1:03d}",
            beat_type=beat_type,
            timestamp_range=(ts, ts + 5),  # 5-second window
            visual_description=mc.visual_description,
            key_dialogue=mc.key_dialogue,
            screenshot_candidates_s=screenshot_candidates,
            emotional_beat=mc.emotional_beat,
        ))
    
    return converted


def _convert_characters(characters: list[CharacterProfile]) -> list[Character]:
    """Convert CharacterProfile objects to Character objects."""
    return [
        Character(
            name=cp.name,
            role=cp.role,
            traits=cp.traits,
            appearance=cp.appearance,
            pronoun=cp.pronoun,
        )
        for cp in characters
    ]

