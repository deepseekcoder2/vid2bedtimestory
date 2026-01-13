"""
Character Signal Analysis

Analyzes subtitle usage patterns to determine if a name refers to
a real character vs a group/object/false positive.

Uses statistical and linguistic signals:
- Frequency of mentions
- Attributed dialogue lines
- Singular vs plural usage
- Individual action context
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from vid2bedtimestory.models import SubtitleSegment


@dataclass
class CharacterSignals:
    """
    Collected signals about a potential character name.
    """
    name: str
    
    # Raw counts
    mention_count: int = 0
    dialogue_count: int = 0  # Lines where name is speaker: "[Name] said..."
    
    # Contextual signals
    plural_usage_count: int = 0  # "Hey campers!" patterns
    singular_usage_count: int = 0  # "Coop said..." patterns
    individual_action_count: int = 0  # "Name grabbed/ran/jumped..."
    collective_action_count: int = 0  # "The campers cheered..."
    
    # Computed
    is_likely_plural: bool = False
    
    # Example contexts for debugging
    example_dialogues: list[str] = field(default_factory=list)
    example_actions: list[str] = field(default_factory=list)


def analyze_character_signals(
    name: str,
    subtitles: list[SubtitleSegment],
) -> CharacterSignals:
    """
    Analyze subtitle usage patterns for a potential character name.
    
    Args:
        name: The name to analyze
        subtitles: List of subtitle segments
        
    Returns:
        CharacterSignals with all collected data
    """
    signals = CharacterSignals(name=name)
    name_lower = name.lower()
    
    for seg in subtitles:
        text = seg.text
        text_lower = text.lower()
        
        # Skip if name not in this segment
        if name_lower not in text_lower:
            continue
        
        signals.mention_count += 1
        
        # Check for dialogue attribution: [Name], (Name), Name:
        if _is_dialogue_attribution(name, text):
            signals.dialogue_count += 1
            if len(signals.example_dialogues) < 3:
                signals.example_dialogues.append(text[:100])
        
        # Check for plural context
        if _is_plural_context(name, text):
            signals.plural_usage_count += 1
        else:
            signals.singular_usage_count += 1
        
        # Check for individual actions
        if _has_individual_action(name, text):
            signals.individual_action_count += 1
            if len(signals.example_actions) < 3:
                signals.example_actions.append(text[:100])
        elif _has_collective_action(name, text):
            signals.collective_action_count += 1
    
    # Compute derived signals
    total_usage = signals.plural_usage_count + signals.singular_usage_count
    if total_usage > 0:
        signals.is_likely_plural = (
            signals.plural_usage_count / total_usage > 0.5
        )
    
    return signals


def score_character_likelihood(signals: CharacterSignals) -> float:
    """
    Score how likely this name refers to a real character.
    
    Returns:
        0.0 = definitely not a character
        1.0 = definitely a character
        
    Thresholds:
        >= 0.7: High confidence character
        <= 0.3: High confidence NOT a character
        0.3-0.7: Uncertain, needs VLM verification
    """
    score = 0.5  # Start neutral
    
    # === POSITIVE SIGNALS ===
    
    # Dialogue attribution is the STRONGEST signal
    # Characters speak. Groups don't have attributed dialogue.
    if signals.dialogue_count >= 5:
        score += 0.4  # Very strong
    elif signals.dialogue_count >= 3:
        score += 0.35
    elif signals.dialogue_count >= 1:
        score += 0.2
    
    # Frequency matters, but less than dialogue
    if signals.mention_count >= 10:
        score += 0.15
    elif signals.mention_count >= 5:
        score += 0.1
    elif signals.mention_count == 1:
        score -= 0.1  # Single mention is weak
    
    # Individual actions suggest a character
    if signals.individual_action_count >= 3:
        score += 0.15
    elif signals.individual_action_count >= 1:
        score += 0.1
    
    # === NEGATIVE SIGNALS ===
    
    # Plural usage is a strong negative signal
    if signals.is_likely_plural:
        score -= 0.3
    
    # Collective actions suggest a group
    if signals.collective_action_count > signals.individual_action_count:
        score -= 0.15
    
    # Dialogue without individual actions is slightly suspicious
    # (could be a narrator or group voice)
    if signals.dialogue_count > 0 and signals.individual_action_count == 0:
        score -= 0.05
    
    return max(0.0, min(1.0, score))


def _is_dialogue_attribution(name: str, text: str) -> bool:
    """
    Check if the name appears as a dialogue speaker.
    
    Patterns:
        [Name] said something
        (Name) said something
        Name: said something
    """
    name_escaped = re.escape(name)
    patterns = [
        rf'\[{name_escaped}\]',  # [Name]
        rf'\({name_escaped}\)',  # (Name)
        rf'^{name_escaped}:',    # Name: (at start)
        rf'\n{name_escaped}:',   # Name: (after newline)
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def _is_plural_context(name: str, text: str) -> bool:
    """
    Check if the name is used in a plural/collective context.
    
    Patterns:
        "Hey campers!"
        "the campers"
        "all campers"
        "campers cheered"
    """
    name_lower = name.lower()
    text_lower = text.lower()
    
    # Check for plural indicators around the name
    plural_patterns = [
        rf'\bhey\s+{re.escape(name_lower)}',      # "Hey campers"
        rf'\bthe\s+{re.escape(name_lower)}',      # "the campers"
        rf'\ball\s+{re.escape(name_lower)}',      # "all campers"
        rf'\bother\s+{re.escape(name_lower)}',    # "other campers"
        rf'\bfellow\s+{re.escape(name_lower)}',   # "fellow campers"
        rf'{re.escape(name_lower)}\s+all\b',      # "campers all"
        rf'{re.escape(name_lower)}\s+were\b',     # "campers were" (plural verb)
        rf'{re.escape(name_lower)}\s+are\b',      # "campers are" (plural verb)
    ]
    
    for pattern in plural_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def _has_individual_action(name: str, text: str) -> bool:
    """
    Check if the name is the subject of an individual action.
    
    Patterns:
        "Coop grabbed the wheel"
        "Uncle Larry walked in"
        "Dash smiled"
    """
    name_escaped = re.escape(name)
    
    # Common action verbs for characters
    action_verbs = [
        'said', 'asked', 'replied', 'shouted', 'whispered', 'yelled', 'called',
        'grabbed', 'took', 'held', 'picked', 'dropped', 'threw', 'caught',
        'ran', 'walked', 'jumped', 'climbed', 'fell', 'stood', 'sat',
        'looked', 'saw', 'watched', 'stared', 'glanced', 'noticed',
        'smiled', 'laughed', 'cried', 'frowned', 'grinned', 'nodded',
        'turned', 'faced', 'pointed', 'reached', 'touched', 'pushed', 'pulled',
        'thought', 'knew', 'felt', 'wanted', 'needed', 'tried', 'decided',
        'is', 'was', 'has', 'had',  # State verbs for "Coop is brave"
    ]
    
    # Build pattern: Name + action verb
    verb_pattern = '|'.join(action_verbs)
    pattern = rf'\b{name_escaped}\s+({verb_pattern})\b'
    
    if re.search(pattern, text, re.IGNORECASE):
        return True
    
    return False


def _has_collective_action(name: str, text: str) -> bool:
    """
    Check if the name is used with collective/group actions.
    
    Patterns:
        "The campers cheered"
        "Campers gathered around"
    """
    name_lower = name.lower()
    text_lower = text.lower()
    
    collective_patterns = [
        rf'the\s+{re.escape(name_lower)}\s+\w+ed',  # "the campers cheered"
        rf'{re.escape(name_lower)}\s+gathered',
        rf'{re.escape(name_lower)}\s+crowded',
        rf'{re.escape(name_lower)}\s+watched',
        rf'{re.escape(name_lower)}\s+cheered',
        rf'{re.escape(name_lower)}\s+laughed',  # Group laughing
        rf'{re.escape(name_lower)}\s+clapped',
    ]
    
    for pattern in collective_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

@dataclass
class CharacterCandidate:
    """
    A potential character with analysis results.
    """
    name: str
    signals: CharacterSignals
    score: float
    decision: str  # "CHARACTER", "NOT_CHARACTER", "UNCERTAIN"


def analyze_all_candidates(
    names: list[str],
    subtitles: list[SubtitleSegment],
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
) -> list[CharacterCandidate]:
    """
    Analyze all candidate names and categorize them.
    
    Args:
        names: List of potential character names
        subtitles: Subtitle segments for analysis
        high_threshold: Score above this = definite character
        low_threshold: Score below this = definite non-character
        
    Returns:
        List of CharacterCandidate with decisions
    """
    candidates = []
    
    for name in names:
        signals = analyze_character_signals(name, subtitles)
        score = score_character_likelihood(signals)
        
        if score >= high_threshold:
            decision = "CHARACTER"
        elif score <= low_threshold:
            decision = "NOT_CHARACTER"
        else:
            decision = "UNCERTAIN"
        
        candidates.append(CharacterCandidate(
            name=name,
            signals=signals,
            score=score,
            decision=decision,
        ))
    
    # Sort by score (highest first)
    candidates.sort(key=lambda c: c.score, reverse=True)
    
    return candidates


def filter_character_names(
    names: list[str],
    subtitles: list[SubtitleSegment],
    include_uncertain: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Filter names into characters, non-characters, and uncertain.
    
    Args:
        names: List of potential character names
        subtitles: Subtitle segments for analysis
        include_uncertain: If True, uncertain names go to characters
        
    Returns:
        Tuple of (character_names, uncertain_names, rejected_names)
    """
    candidates = analyze_all_candidates(names, subtitles)
    
    characters = []
    uncertain = []
    rejected = []
    
    for c in candidates:
        if c.decision == "CHARACTER":
            characters.append(c.name)
        elif c.decision == "NOT_CHARACTER":
            rejected.append(c.name)
        else:  # UNCERTAIN
            uncertain.append(c.name)
    
    return characters, uncertain, rejected

