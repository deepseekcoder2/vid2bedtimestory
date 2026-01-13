"""
Phase 4: Character Extraction

Extract character profiles with explicit appearance and pronouns.

Supports two modes:
1. FRANCHISE MODE: Use character database as source of truth
2. DISCOVERY MODE: Use signal analysis + multi-frame VLM consensus

The franchise database provides:
- Guaranteed correct pronouns
- Visual signatures for VLM matching
- Catchphrases and traits for story generation
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from vid2bedtimestory.models import SubtitleSegment
from vid2bedtimestory.knowledge import FranchiseData, CharacterData

from ..types import MomentCaption, CharacterProfile, FrameExtractionError, VLMError
from ..prompts import CHARACTER_PROMPT
from ..config import get_config
from ..frame_utils import extract_frame_cached
from ..subtitle_utils import extract_character_names_from_subtitles, find_character_speech
from ..vlm_client import caption_frame
from ..character_signals import (
    analyze_all_candidates,
    CharacterCandidate,
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def extract_characters(
    video_path: Path,
    moments: list[MomentCaption],
    subtitles: list[SubtitleSegment],
    franchise_db: Optional[FranchiseData] = None,
    max_characters: int = None,
    frames_per_character: int = 3,
) -> list[CharacterProfile]:
    """
    Extract character profiles using franchise DB or discovery.
    
    If franchise_db is provided:
        - Characters from DB are used directly (correct pronouns guaranteed)
        - Unknown names go through signal analysis + VLM discovery
        
    If franchise_db is None:
        - All names go through signal analysis + VLM discovery
    
    Args:
        video_path: Path to the video file
        moments: List of MomentCaption from deep dive phase
        subtitles: List of SubtitleSegment with dialogue
        franchise_db: Optional franchise database for known characters
        max_characters: Maximum characters to extract
        frames_per_character: Number of frames for VLM discovery
        
    Returns:
        List of CharacterProfile objects
    """
    config = get_config()
    max_characters = max_characters or config.max_characters
    
    # Step 1: Extract all candidate names from subtitles
    raw_names = extract_character_names_from_subtitles(subtitles)
    print(f"[character_extraction] Found {len(raw_names)} raw name candidates")
    
    # Step 2: Route names based on franchise DB
    if franchise_db:
        profiles = _extract_with_franchise_db(
            video_path=video_path,
            moments=moments,
            subtitles=subtitles,
            raw_names=raw_names,
            franchise_db=franchise_db,
            max_characters=max_characters,
            frames_per_character=frames_per_character,
        )
    else:
        profiles = _extract_with_discovery(
            video_path=video_path,
            moments=moments,
            subtitles=subtitles,
            raw_names=raw_names,
            max_characters=max_characters,
            frames_per_character=frames_per_character,
        )
    
    return profiles[:max_characters]


# =============================================================================
# FRANCHISE MODE
# =============================================================================

def _extract_with_franchise_db(
    video_path: Path,
    moments: list[MomentCaption],
    subtitles: list[SubtitleSegment],
    raw_names: list[str],
    franchise_db: FranchiseData,
    max_characters: int,
    frames_per_character: int,
) -> list[CharacterProfile]:
    """
    Extract characters using franchise database as source of truth.
    """
    print(f"[character_extraction] Using franchise DB: {franchise_db.franchise_name}")
    
    profiles: list[CharacterProfile] = []
    discovered_names: list[str] = []
    
    for name in raw_names:
        # Check if known non-character
        if franchise_db.is_known_non_character(name):
            print(f"[character_extraction]   ✗ {name}: known non-character (skip)")
            continue
        
        # Try to find in database
        char_data = franchise_db.get_character(name)
        
        if char_data:
            # Found in DB - use database info
            profile = _character_from_db(char_data, video_path, moments, subtitles)
            if profile:
                profiles.append(profile)
                print(f"[character_extraction]   ✓ {name}: from DB ({char_data.pronoun})")
        else:
            # Not in DB - queue for discovery
            discovered_names.append(name)
    
    # Process unknown names through signal analysis + discovery
    if discovered_names and len(profiles) < max_characters:
        remaining_slots = max_characters - len(profiles)
        discovered = _extract_with_discovery(
            video_path=video_path,
            moments=moments,
            subtitles=subtitles,
            raw_names=discovered_names,
            max_characters=remaining_slots,
            frames_per_character=frames_per_character,
        )
        profiles.extend(discovered)
    
    return profiles


def _character_from_db(
    char_data: CharacterData,
    video_path: Path,
    moments: list[MomentCaption],
    subtitles: list[SubtitleSegment],
) -> Optional[CharacterProfile]:
    """
    Create CharacterProfile from database entry.
    
    Finds a reference frame using visual signature for VLM confirmation.
    """
    # Find reference frame
    frame_path, timestamp_s = _find_character_frame_with_signature(
        video_path=video_path,
        character_name=char_data.display_name,
        visual_signature=char_data.visual_signature,
        moments=moments,
        subtitles=subtitles,
    )
    
    if frame_path is None:
        # Fall back to speech timestamps
        speech_instances = find_character_speech(subtitles, char_data.display_name)
        if speech_instances:
            timestamp_s = speech_instances[0][0]
            try:
                frame_path = extract_frame_cached(video_path, timestamp_s)
            except FrameExtractionError:
                pass
    
    if frame_path is None and moments:
        frame_path = moments[0].frame_path
        timestamp_s = moments[0].timestamp_s
    
    # Normalize pronoun format
    pronoun = char_data.pronoun
    if "/" in pronoun:
        pronoun = pronoun.split("/")[0]  # "he/him" -> "he"
    
    return CharacterProfile(
        name=char_data.display_name,
        role=char_data.role,
        traits=char_data.traits[:4],
        appearance=_format_visual_signature(char_data.visual_signature),
        pronoun=pronoun,
        first_appearance_s=timestamp_s or 0.0,
        reference_frame_path=frame_path or Path("."),
    )


def _find_character_frame_with_signature(
    video_path: Path,
    character_name: str,
    visual_signature: dict,
    moments: list[MomentCaption],
    subtitles: list[SubtitleSegment],
) -> tuple[Optional[Path], float]:
    """
    Find a frame where character appears using visual signature.
    
    Uses VLM to verify character presence based on signature.
    """
    name_lower = character_name.lower()
    
    # First try: moments that mention character by name
    for moment in moments:
        if name_lower in moment.visual_description.lower():
            return moment.frame_path, moment.timestamp_s
    
    # Second try: moments near character's dialogue
    speech_instances = find_character_speech(subtitles, character_name)
    for speech_ts, _ in speech_instances[:3]:
        nearest = _find_nearest_moment(moments, speech_ts)
        if nearest:
            return nearest.frame_path, nearest.timestamp_s
    
    # Fall back to first moment
    if moments:
        return moments[0].frame_path, moments[0].timestamp_s
    
    return None, 0.0


def _format_visual_signature(sig: dict) -> str:
    """Format visual signature dict as readable appearance string."""
    parts = []
    
    if sig.get("age"):
        parts.append(sig["age"])
    if sig.get("hair"):
        parts.append(f"hair: {sig['hair']}")
    if sig.get("skin_tone"):
        parts.append(f"skin tone: {sig['skin_tone']}")
    if sig.get("clothing"):
        parts.append(f"wearing {sig['clothing']}")
    if sig.get("distinctive"):
        parts.append(sig["distinctive"])
    
    return ". ".join(parts) if parts else ""


# =============================================================================
# DISCOVERY MODE
# =============================================================================

def _extract_with_discovery(
    video_path: Path,
    moments: list[MomentCaption],
    subtitles: list[SubtitleSegment],
    raw_names: list[str],
    max_characters: int,
    frames_per_character: int,
) -> list[CharacterProfile]:
    """
    Extract characters using signal analysis + VLM discovery.
    """
    print(f"[character_extraction] Discovery mode for {len(raw_names)} names")
    
    # Step 1: Signal analysis to filter candidates
    candidates = analyze_all_candidates(raw_names, subtitles)
    
    # Log analysis results
    for c in candidates:
        status = "✓" if c.decision == "CHARACTER" else "?" if c.decision == "UNCERTAIN" else "✗"
        print(f"[character_extraction]   {status} {c.name}: score={c.score:.2f} ({c.decision})")
    
    # Step 2: Process high-confidence + uncertain candidates
    profiles: list[CharacterProfile] = []
    
    for candidate in candidates:
        if len(profiles) >= max_characters:
            break
            
        if candidate.decision == "NOT_CHARACTER":
            continue
        
        # Use stricter VLM verification for uncertain candidates
        require_strict = (candidate.decision == "UNCERTAIN")
        
        try:
            profile = _extract_character_with_consensus(
                video_path=video_path,
                character_name=candidate.name,
                moments=moments,
                subtitles=subtitles,
                num_frames=frames_per_character,
                strict_validation=require_strict,
            )
            if profile:
                profiles.append(profile)
                print(f"[character_extraction]   ✓ {candidate.name}: VLM confirmed ({profile.pronoun})")
        except (FrameExtractionError, VLMError) as e:
            print(f"[character_extraction]   ✗ {candidate.name}: VLM error ({e})")
            continue
    
    return profiles


# =============================================================================
# VLM MULTI-FRAME CONSENSUS
# =============================================================================

def _extract_character_with_consensus(
    video_path: Path,
    character_name: str,
    moments: list[MomentCaption],
    subtitles: list[SubtitleSegment],
    num_frames: int = 3,
    strict_validation: bool = False,
) -> Optional[CharacterProfile]:
    """
    Extract profile using multi-frame consensus.
    
    If strict_validation=True, requires stronger agreement between frames.
    """
    # Find multiple candidate frames
    frame_candidates = _find_character_frames(
        video_path=video_path,
        character_name=character_name,
        moments=moments,
        subtitles=subtitles,
        max_frames=num_frames,
    )
    
    if not frame_candidates:
        return None
    
    # Extract profile from each frame independently
    observations: list[dict] = []
    first_timestamp = None
    first_frame_path = None
    
    for frame_path, timestamp_s in frame_candidates:
        if first_timestamp is None:
            first_timestamp = timestamp_s
            first_frame_path = frame_path
        
        try:
            prompt = CHARACTER_PROMPT.format(name=character_name)
            response = caption_frame(frame_path, prompt)
            parsed = _parse_character_response(response)
            
            if _is_valid_character_observation(parsed):
                observations.append(parsed)
        except VLMError:
            continue
    
    if not observations:
        return None
    
    # Strict validation requires multiple confirming observations
    if strict_validation and len(observations) < 2:
        return None
    
    # Build consensus from observations
    consensus = _build_consensus(observations)
    
    if consensus is None:
        return None
    
    return CharacterProfile(
        name=character_name,
        role=consensus["role"],
        traits=consensus["traits"],
        appearance=consensus["appearance"],
        pronoun=consensus["pronoun"],
        first_appearance_s=first_timestamp or 0.0,
        reference_frame_path=first_frame_path or Path("."),
    )


def _find_character_frames(
    video_path: Path,
    character_name: str,
    moments: list[MomentCaption],
    subtitles: list[SubtitleSegment],
    max_frames: int = 3,
) -> list[tuple[Path, float]]:
    """Find multiple reference frames where character might appear."""
    name_lower = character_name.lower()
    frames: list[tuple[Path, float]] = []
    seen_timestamps: set[float] = set()
    
    # Strategy 1: Moments that mention character
    for moment in moments:
        if name_lower in moment.visual_description.lower():
            if moment.timestamp_s not in seen_timestamps:
                frames.append((moment.frame_path, moment.timestamp_s))
                seen_timestamps.add(moment.timestamp_s)
                if len(frames) >= max_frames:
                    return frames
    
    # Strategy 2: Speech timestamps
    speech_instances = find_character_speech(subtitles, character_name)
    for speech_ts, _ in speech_instances[:5]:
        nearest_moment = _find_nearest_moment(moments, speech_ts)
        
        if nearest_moment and nearest_moment.timestamp_s not in seen_timestamps:
            frames.append((nearest_moment.frame_path, nearest_moment.timestamp_s))
            seen_timestamps.add(nearest_moment.timestamp_s)
        elif speech_ts not in seen_timestamps:
            try:
                frame_path = extract_frame_cached(video_path, speech_ts)
                frames.append((frame_path, speech_ts))
                seen_timestamps.add(speech_ts)
            except FrameExtractionError:
                continue
        
        if len(frames) >= max_frames:
            return frames
    
    # Strategy 3: Early moments (protagonist often visible early)
    for moment in moments[:10]:
        if moment.timestamp_s not in seen_timestamps:
            frames.append((moment.frame_path, moment.timestamp_s))
            seen_timestamps.add(moment.timestamp_s)
            if len(frames) >= max_frames:
                return frames
    
    return frames


def _find_nearest_moment(
    moments: list[MomentCaption],
    target_ts: float,
) -> Optional[MomentCaption]:
    """Find the moment nearest to a target timestamp."""
    if not moments:
        return None
    
    nearest = min(moments, key=lambda m: abs(m.timestamp_s - target_ts))
    
    if abs(nearest.timestamp_s - target_ts) <= 10.0:
        return nearest
    
    return None


def _is_valid_character_observation(parsed: dict) -> bool:
    """Check if VLM response describes a valid human character."""
    appearance = parsed.get("appearance", "").lower()
    
    # Reject vehicle descriptions
    vehicle_keywords = [
        "car", "vehicle", "automobile", "truck", "race car",
        "wheels", "engine", "exhaust", "bumper", "headlight",
        "chassis", "spoiler", "hood", "tire",
    ]
    
    for keyword in vehicle_keywords:
        if f"is a {keyword}" in appearance or f"is an {keyword}" in appearance:
            return False
        if appearance.startswith(keyword):
            return False
    
    # Require human-like descriptors
    human_keywords = [
        "hair", "eyes", "skin", "wearing", "clothes", "shirt", "jacket",
        "pants", "dress", "face", "smile", "expression", "boy", "girl",
        "man", "woman", "child", "teen", "adult", "person",
    ]
    
    return any(kw in appearance for kw in human_keywords)


def _build_consensus(observations: list[dict]) -> Optional[dict]:
    """Build consensus profile from multiple VLM observations."""
    if not observations:
        return None
    
    if len(observations) == 1:
        return observations[0]
    
    # Pronoun consensus (majority vote)
    pronouns = [obs["pronoun"] for obs in observations if obs.get("pronoun")]
    pronoun_counts = Counter(pronouns)
    
    if pronoun_counts:
        most_common = pronoun_counts.most_common(1)
        pronoun = most_common[0][0]
    else:
        pronoun = "they"
    
    # Role consensus
    roles = [obs["role"] for obs in observations if obs.get("role")]
    role_counts = Counter(roles)
    role = role_counts.most_common(1)[0][0] if role_counts else "neutral"
    
    # Traits synthesis
    all_traits = []
    for obs in observations:
        all_traits.extend(obs.get("traits", []))
    
    trait_counts = Counter(all_traits)
    if len(observations) >= 2:
        consensus_traits = [t for t, c in trait_counts.items() if c >= 2]
        if not consensus_traits:
            consensus_traits = [t for t, _ in trait_counts.most_common(4)]
    else:
        consensus_traits = list(trait_counts.keys())[:4]
    
    # Appearance (use longest description)
    appearances = [obs.get("appearance", "") for obs in observations if obs.get("appearance")]
    appearance = max(appearances, key=len) if appearances else ""
    
    return {
        "pronoun": pronoun,
        "role": role,
        "traits": consensus_traits[:4],
        "appearance": appearance,
    }


def _parse_character_response(response: str) -> dict:
    """Parse VLM response for character extraction."""
    try:
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        
        role = data.get("role", "neutral").lower()
        valid_roles = {"protagonist", "ally", "mentor", "antagonist", "neutral", "villain", "recurring"}
        if role not in valid_roles:
            role = "neutral"
        
        pronoun = data.get("pronoun", "they").lower()
        valid_pronouns = {"he", "she", "they"}
        if pronoun not in valid_pronouns:
            pronoun = "they"
        
        return {
            "role": role,
            "appearance": data.get("appearance", ""),
            "traits": data.get("traits", [])[:4],
            "pronoun": pronoun,
        }
    except json.JSONDecodeError:
        return {
            "role": "neutral",
            "appearance": response[:200] if response else "",
            "traits": [],
            "pronoun": "they",
        }
