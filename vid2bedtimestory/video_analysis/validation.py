"""
Validation utilities for video analysis outputs.

Validates phase outputs before assembly to catch problems early.
"""

from collections import Counter
from typing import Optional

from .types import BeatCandidate, MomentCaption, CharacterProfile, ValidationError


def validate_beats(beats: list[BeatCandidate], duration_s: float) -> None:
    """
    Ensure beats cover full timeline without gaps/overlaps.
    
    Args:
        beats: List of detected story beats
        duration_s: Total video duration in seconds
        
    Raises:
        ValidationError if:
        - First beat starts >30s into video
        - Last beat ends >30s before video end
        - Gap >60s between consecutive beats
        - Beats overlap
    """
    if not beats:
        raise ValidationError("No beats detected")
    
    # Sort beats by start time
    sorted_beats = sorted(beats, key=lambda b: b.time_range[0])
    
    # Check first beat starts near beginning
    first_start = sorted_beats[0].time_range[0]
    if first_start > 30:
        raise ValidationError(
            f"First beat starts too late: {first_start:.1f}s (should be <30s)"
        )
    
    # Check last beat ends near end
    last_end = sorted_beats[-1].time_range[1]
    if last_end < duration_s - 30:
        raise ValidationError(
            f"Last beat ends too early: {last_end:.1f}s (video is {duration_s:.1f}s)"
        )
    
    # Check for gaps and overlaps between consecutive beats
    for i in range(len(sorted_beats) - 1):
        current_end = sorted_beats[i].time_range[1]
        next_start = sorted_beats[i + 1].time_range[0]
        
        # Check for gap
        gap = next_start - current_end
        if gap > 60:
            raise ValidationError(
                f"Gap of {gap:.1f}s between beats {i+1} and {i+2} "
                f"({current_end:.1f}s to {next_start:.1f}s)"
            )
        
        # Check for overlap (allow 1s tolerance)
        if current_end > next_start + 1:
            raise ValidationError(
                f"Beats {i+1} and {i+2} overlap: beat {i+1} ends at {current_end:.1f}s "
                f"but beat {i+2} starts at {next_start:.1f}s"
            )


def validate_moments(moments: list[MomentCaption], duration_s: float) -> None:
    """
    Ensure moments have good coverage and quality.
    
    Args:
        moments: List of moment captions
        duration_s: Total video duration in seconds
        
    Raises:
        ValidationError if:
        - Fewer than 20 moments
        - >10% duplicate visual_descriptions
        - Moments don't span at least 80% of video duration
    """
    min_required = max(10, int(duration_s / 30))  # ~1 moment per 30 seconds
    if len(moments) < min_required:
        raise ValidationError(
            f"Too few moments: {len(moments)} (minimum {min_required} required for {duration_s:.0f}s video)"
        )
    
    # Check for duplicates
    descriptions = [m.visual_description for m in moments]
    description_counts = Counter(descriptions)
    duplicates = sum(count - 1 for count in description_counts.values() if count > 1)
    duplicate_ratio = duplicates / len(moments)
    
    if duplicate_ratio > 0.1:
        raise ValidationError(
            f"Too many duplicate descriptions: {duplicate_ratio:.1%} "
            f"({duplicates} duplicates out of {len(moments)} moments)"
        )
    
    # Check timeline coverage
    if moments:
        timestamps = [m.timestamp_s for m in moments]
        span = max(timestamps) - min(timestamps)
        coverage = span / duration_s
        
        if coverage < 0.8:
            raise ValidationError(
                f"Poor timeline coverage: moments span only {coverage:.1%} of video "
                f"({min(timestamps):.1f}s to {max(timestamps):.1f}s out of {duration_s:.1f}s)"
            )


def validate_characters(characters: list[CharacterProfile]) -> None:
    """
    Ensure characters have required fields.
    
    Args:
        characters: List of character profiles
        
    Raises:
        ValidationError if:
        - Any character missing appearance
        - Any character missing pronoun
    """
    for char in characters:
        if not char.appearance:
            raise ValidationError(
                f"Character '{char.name}' missing appearance description"
            )
        
        if not char.pronoun:
            raise ValidationError(
                f"Character '{char.name}' missing pronoun"
            )


def validate_all(
    beats: list[BeatCandidate],
    moments: list[MomentCaption],
    characters: list[CharacterProfile],
    duration_s: float,
) -> list[str]:
    """
    Run all validations, return list of warning messages.
    
    Critical failures raise ValidationError.
    Non-critical issues are returned as warnings.
    
    Args:
        beats: List of detected story beats
        moments: List of moment captions
        characters: List of character profiles
        duration_s: Total video duration in seconds
        
    Returns:
        List of warning messages for non-critical issues
        
    Raises:
        ValidationError: For critical failures that should stop processing
    """
    warnings: list[str] = []
    
    # Validate beats (critical)
    try:
        validate_beats(beats, duration_s)
    except ValidationError as e:
        # Re-raise - beat validation is critical
        raise
    
    # Validate moments
    try:
        validate_moments(moments, duration_s)
    except ValidationError as e:
        # Moment validation can be a warning if we have some moments
        if len(moments) >= 10:
            warnings.append(f"Moment validation warning: {e}")
        else:
            raise
    
    # Validate characters (warning only - may have no attributed dialogue)
    try:
        validate_characters(characters)
    except ValidationError as e:
        warnings.append(f"Character validation warning: {e}")
    
    # Additional non-critical checks
    
    # Check beat type distribution
    beat_types = [b.beat_type for b in beats]
    if "climax" not in beat_types:
        warnings.append("No climax beat detected - story structure may be unclear")
    
    if "inciting_incident" not in beat_types:
        warnings.append("No inciting_incident beat detected - story structure may be unclear")
    
    # Check moment distribution across beats
    beat_ids = set(b.beat_id for b in beats)
    moment_beat_ids = set(m.beat_id for m in moments)
    uncovered_beats = beat_ids - moment_beat_ids
    
    if uncovered_beats:
        warnings.append(
            f"{len(uncovered_beats)} beats have no moments: {sorted(uncovered_beats)}"
        )
    
    # Check character count
    if len(characters) == 0:
        warnings.append("No characters extracted - dialogue may lack attribution")
    
    return warnings

