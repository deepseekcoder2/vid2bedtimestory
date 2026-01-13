"""
Subtitle utilities for video analysis.

Provides functions to query subtitles by time range and find anchor points.
"""

from typing import Optional
import re

from vid2bedtimestory.models import SubtitleSegment


def get_subtitles_in_range(
    subtitles: list[SubtitleSegment],
    start_s: float,
    end_s: float,
    include_partial: bool = True,
) -> list[SubtitleSegment]:
    """
    Get subtitle segments that fall within a time range.
    
    Args:
        subtitles: List of subtitle segments
        start_s: Start time in seconds
        end_s: End time in seconds
        include_partial: If True, include segments that partially overlap
        
    Returns:
        List of matching SubtitleSegment objects
    """
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)
    
    results = []
    for seg in subtitles:
        if include_partial:
            # Include if any overlap
            if seg.end_ms >= start_ms and seg.start_ms <= end_ms:
                results.append(seg)
        else:
            # Include only if fully contained
            if seg.start_ms >= start_ms and seg.end_ms <= end_ms:
                results.append(seg)
    
    return results


def get_dialogue_texts_in_range(
    subtitles: list[SubtitleSegment],
    start_s: float,
    end_s: float,
    exclude_sound_effects: bool = True,
) -> list[str]:
    """
    Get dialogue text strings within a time range.
    
    Args:
        subtitles: List of subtitle segments
        start_s: Start time in seconds
        end_s: End time in seconds
        exclude_sound_effects: If True, exclude [bracketed] sound effects
        
    Returns:
        List of dialogue text strings
    """
    segments = get_subtitles_in_range(subtitles, start_s, end_s)
    
    texts = []
    for seg in segments:
        text = seg.text.strip()
        
        if exclude_sound_effects:
            # Skip pure sound effects like "[engine revs]" or "[music playing]"
            if re.match(r'^\[.*\]$', text):
                continue
            # Remove inline sound effects but keep speech
            text = re.sub(r'\[.*?\]', '', text).strip()
            if not text:
                continue
        
        texts.append(text)
    
    return texts


def get_dialogue_timestamps(
    subtitles: list[SubtitleSegment],
    start_s: float,
    end_s: float,
    exclude_sound_effects: bool = True,
) -> list[float]:
    """
    Get timestamps where dialogue occurs within a range.
    
    Returns the start time of each dialogue segment.
    
    Args:
        subtitles: List of subtitle segments
        start_s: Start time in seconds
        end_s: End time in seconds
        exclude_sound_effects: If True, exclude [bracketed] sound effects
        
    Returns:
        List of timestamps in seconds
    """
    segments = get_subtitles_in_range(subtitles, start_s, end_s)
    
    timestamps = []
    for seg in segments:
        if exclude_sound_effects:
            # Skip pure sound effects
            if re.match(r'^\[.*\]$', seg.text.strip()):
                continue
        
        timestamps.append(seg.start_ms / 1000.0)
    
    return timestamps


def find_dialogue_anchor(
    subtitles: list[SubtitleSegment],
    query: str,
    fuzzy: bool = True,
) -> Optional[float]:
    """
    Find timestamp where dialogue matches a query.
    
    Args:
        subtitles: List of subtitle segments
        query: Text to search for
        fuzzy: If True, match partial/case-insensitive
        
    Returns:
        Timestamp in seconds where match found, or None
    """
    query_lower = query.lower()
    
    for seg in subtitles:
        text = seg.text.strip()
        
        if fuzzy:
            if query_lower in text.lower():
                return seg.start_ms / 1000.0
        else:
            if query == text:
                return seg.start_ms / 1000.0
    
    return None


def find_character_speech(
    subtitles: list[SubtitleSegment],
    character_name: str,
) -> list[tuple[float, str]]:
    """
    Find all instances where a character speaks.
    
    Looks for patterns like "[CharacterName] text" or "(CharacterName) text".
    
    Args:
        subtitles: List of subtitle segments
        character_name: Name to search for
        
    Returns:
        List of (timestamp_s, dialogue_text) tuples
    """
    results = []
    name_lower = character_name.lower()
    
    # Patterns to match character attribution
    patterns = [
        rf'\[{re.escape(name_lower)}\]',  # [Coop]
        rf'\({re.escape(name_lower)}\)',  # (Coop)
        rf'^{re.escape(name_lower)}:',    # Coop:
        rf'^{re.escape(name_lower)} ',    # Coop says...
    ]
    
    for seg in subtitles:
        text = seg.text.strip()
        text_lower = text.lower()
        
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Extract the actual dialogue (remove attribution)
                dialogue = re.sub(r'^\[.*?\]\s*', '', text)
                dialogue = re.sub(r'^\(.*?\)\s*', '', dialogue)
                dialogue = re.sub(rf'^{re.escape(character_name)}:\s*', '', dialogue, flags=re.IGNORECASE)
                
                if dialogue:
                    results.append((seg.start_ms / 1000.0, dialogue.strip()))
                break
    
    return results


def format_subtitles_for_llm(
    subtitles: list[SubtitleSegment],
    include_timestamps: bool = True,
    max_segments: Optional[int] = None,
) -> str:
    """
    Format subtitles as text for LLM consumption.
    
    Args:
        subtitles: List of subtitle segments
        include_timestamps: If True, include timestamps
        max_segments: Maximum number of segments to include
        
    Returns:
        Formatted string
    """
    lines = []
    
    segments_to_format = subtitles
    if max_segments:
        segments_to_format = subtitles[:max_segments]
    
    for seg in segments_to_format:
        if include_timestamps:
            ts = seg.start_ms / 1000.0
            lines.append(f"[{ts:.1f}s] {seg.text}")
        else:
            lines.append(seg.text)
    
    if max_segments and len(subtitles) > max_segments:
        lines.append(f"... ({len(subtitles) - max_segments} more lines)")
    
    return "\n".join(lines)


def extract_character_names_from_subtitles(
    subtitles: list[SubtitleSegment],
) -> list[str]:
    """
    Extract character names mentioned in subtitle attributions.
    
    Looks for patterns like [CharacterName], (CharacterName), CharacterName:
    
    Args:
        subtitles: List of subtitle segments
        
    Returns:
        List of unique character names found
    """
    names = set()
    
    # Patterns for character attribution
    patterns = [
        r'\[([A-Z][a-z\']+(?:\s+[A-Z][a-z\']+)?)\]',  # [Name] or [First Last]
        r'\(([A-Z][a-z\']+(?:\s+[A-Z][a-z\']+)?)\)',  # (Name)
        r'^([A-Z][a-z\']+(?:\s+[A-Z][a-z\']+)?):',    # Name:
    ]
    
    # Common false positives to exclude
    exclude = {
        # Single-word exclamations/interjections
        'the', 'a', 'an', 'oh', 'ah', 'uh', 'um', 'wow', 'yeah', 'yes', 'no',
        'hey', 'whoa', 'yay', 'ow', 'ooh', 'aah', 'hmm', 'huh',
        # Group references (not individual characters)
        'campers', 'everyone', 'all', 'guys', 'kids', 'team', 'friends',
        'crowd', 'audience', 'people', 'others', 'both',
        # Sound effect markers
        'sfx', 'music', 'sounds', 'noise',
        # Generic descriptors
        'narrator', 'announcer', 'reporter', 'voice', 'speaker',
        'breaking', 'news', 'live',
    }
    
    for seg in subtitles:
        text = seg.text.strip()
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Filter out common false positives
                if match.lower() not in exclude:
                    names.add(match)
    
    return sorted(names)

