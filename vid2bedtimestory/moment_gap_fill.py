"""
Moment Gap Fill - VideoAgent-style iterative retrieval.

After story is written, this module:
1. Extracts key action phrases from story paragraphs
2. Checks if matching moments exist in the analysis
3. Retrieves missing frames using semantic search + subtitle hints
4. Captions new frames and adds them to the analysis

This follows the VideoAgent (ECCV 2024) philosophy: retrieve only what's needed,
guided by what the story tells us is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .config import config
from .llm import call_llm, LLMError
from .models import AnalysisResult, SubtitleSegment, Moment
from .embedding.index import FrameIndex, load_frame_index
from .embedding.search import semantic_search_frames, SearchResult
from .video_analysis.vlm_client import caption_frame
from .video_analysis.frame_utils import extract_frame_cached
from .video_analysis.prompts import DEEP_DIVE_PROMPT

if TYPE_CHECKING:
    from .knowledge.loader import FranchiseData


@dataclass
class StoryAction:
    """An action phrase extracted from a story paragraph."""
    action_phrase: str
    characters: list[str]
    paragraph_index: int
    source_text: str  # The original paragraph


@dataclass 
class GapFillResult:
    """Result of the gap fill process."""
    actions_extracted: int
    gaps_found: int
    gaps_filled: int
    gaps_unfillable: int
    new_moments: list[Moment]


# =============================================================================
# PROMPT FOR ACTION EXTRACTION
# =============================================================================

ACTION_EXTRACTION_SYSTEM = """You are analyzing a children's story to identify key visual actions that should be illustrated.

For each paragraph, extract the PRIMARY visual action being described. Focus on:
- Character actions (running, jumping, driving, eating, etc.)
- Key moments (discovery, confrontation, celebration)
- Visual transformations (something breaks, changes, appears)

Output JSON format:
{
  "actions": [
    {
      "paragraph_index": 0,
      "action_phrase": "brief description of the visual action (5-10 words)",
      "characters": ["Character1", "Character2"]
    }
  ]
}

Rules:
- One action per paragraph maximum
- Skip paragraphs that are purely dialogue or narration without visual action
- action_phrase should describe WHAT IS VISUALLY HAPPENING, not dialogue
- Keep action_phrase concise but specific enough for visual search
"""

ACTION_EXTRACTION_USER = """Extract the key visual actions from this story:

{story_text}

Return JSON with one action per paragraph that has a clear visual action."""


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def extract_actions_from_story(
    story_text: str,
    franchise_db: Optional["FranchiseData"] = None,
) -> list[StoryAction]:
    """
    Extract key visual actions from story paragraphs.
    
    Args:
        story_text: The full story markdown text
        franchise_db: Optional franchise data for character name hints
        
    Returns:
        List of StoryAction objects
    """
    # Split story into paragraphs
    paragraphs = [p.strip() for p in story_text.split('\n\n') if p.strip()]
    
    # Filter out title and very short paragraphs
    paragraphs = [p for p in paragraphs if len(p) > 50 and not p.startswith('#')]
    
    if not paragraphs:
        return []
    
    # Build numbered paragraph list for the LLM
    numbered_paragraphs = "\n\n".join(
        f"[Paragraph {i}]\n{p}" for i, p in enumerate(paragraphs)
    )
    
    # Call LLM to extract actions
    user_prompt = ACTION_EXTRACTION_USER.format(story_text=numbered_paragraphs)
    
    try:
        response = call_llm(
            messages=[
                {"role": "system", "content": ACTION_EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        
        # Parse JSON response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        data = json.loads(response.strip())
        actions = data.get("actions", [])
        
        # Convert to StoryAction objects
        results = []
        for action in actions:
            idx = action.get("paragraph_index", 0)
            if 0 <= idx < len(paragraphs):
                results.append(StoryAction(
                    action_phrase=action.get("action_phrase", ""),
                    characters=action.get("characters", []),
                    paragraph_index=idx,
                    source_text=paragraphs[idx],
                ))
        
        return results
        
    except (json.JSONDecodeError, LLMError) as e:
        print(f"[gap_fill] WARNING: Failed to extract actions: {e}")
        return []


def find_matching_moment(
    action: StoryAction,
    analysis: AnalysisResult,
    subtitles: list[SubtitleSegment],
    threshold: float = 0.5,
) -> Optional[Moment]:
    """
    Check if an action has a matching moment in the analysis.
    
    Uses text similarity between action phrase and moment descriptions.
    
    Args:
        action: The story action to match
        analysis: Current analysis with moments
        subtitles: Subtitles for timestamp hints
        threshold: Minimum similarity score to consider a match
        
    Returns:
        Matching MomentCaption if found, None otherwise
    """
    action_lower = action.action_phrase.lower()
    action_words = set(action_lower.split())
    
    best_match = None
    best_score = 0.0
    
    for moment in analysis.moments:
        # Simple word overlap scoring
        desc_lower = moment.visual_description.lower()
        desc_words = set(desc_lower.split())
        
        # Check for character names
        char_bonus = 0.0
        for char in action.characters:
            if char.lower() in desc_lower:
                char_bonus += 0.2
        
        # Word overlap
        overlap = len(action_words & desc_words)
        word_score = overlap / max(len(action_words), 1)
        
        # Check key dialogue for hints
        dialogue_bonus = 0.0
        for dialogue in moment.key_dialogue:
            if any(word in dialogue.lower() for word in action_words if len(word) > 3):
                dialogue_bonus += 0.15
        
        total_score = word_score + char_bonus + dialogue_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_match = moment
    
    if best_score >= threshold:
        return best_match
    
    return None


def find_timestamp_from_subtitles(
    action: StoryAction,
    subtitles: list[SubtitleSegment],
) -> Optional[float]:
    """
    Try to find a timestamp hint from subtitles.
    
    Looks for character names or key action words in dialogue.
    
    Args:
        action: The action to find
        subtitles: List of subtitle segments
        
    Returns:
        Timestamp in seconds if found, None otherwise
    """
    action_words = [w.lower() for w in action.action_phrase.split() if len(w) > 3]
    char_names = [c.lower() for c in action.characters]
    
    best_match = None
    best_score = 0
    
    for sub in subtitles:
        text_lower = sub.text.lower()
        
        # Skip sound effects (bracketed text)
        if text_lower.startswith('[') and text_lower.endswith(']'):
            continue
        
        score = 0
        
        # Check for character names
        for char in char_names:
            if char in text_lower:
                score += 2
        
        # Check for action words
        for word in action_words:
            if word in text_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = sub
    
    if best_match and best_score >= 2:
        return best_match.start_ms / 1000.0
    
    return None


def retrieve_and_caption_frame(
    timestamp_s: float,
    video_path: Path,
    action: StoryAction,
    moment_index: int,
    franchise_db: Optional["FranchiseData"] = None,
) -> Optional[Moment]:
    """
    Extract a frame at the given timestamp and caption it.
    
    Args:
        timestamp_s: Timestamp to extract
        video_path: Path to video file
        action: The action we're trying to capture
        moment_index: Index for generating moment_id
        franchise_db: Optional franchise data for character context
        
    Returns:
        Moment if successful, None otherwise
    """
    try:
        # Extract frame
        frame_path = extract_frame_cached(video_path, timestamp_s)
        
        # Build prompt with context
        char_ref = ""
        franchise_examples = ""
        if franchise_db:
            char_lines = []
            for cid, info in franchise_db.characters.items():
                vis = info.visual_signature
                char_lines.append(f"- {info.display_name}: {vis.get('hair', '')}, {vis.get('clothing', '')}")
            char_ref = "CHARACTER REFERENCE:\n" + "\n".join(char_lines)
            franchise_examples = franchise_db.get_deep_dive_examples()
        
        prompt = DEEP_DIVE_PROMPT.format(
            dialogue_context=f"Action context: {action.action_phrase}",
            character_reference=char_ref,
            franchise_deep_dive_examples=franchise_examples,
        )
        
        # Caption with VLM (prefer cloud for quality)
        caption_raw = caption_frame(frame_path, prompt, prefer_cloud=True)
        
        # Parse response
        from .video_analysis.vlm_client import extract_json_robust
        parsed = extract_json_robust(caption_raw)
        
        visual_desc = parsed.get("visual_description", caption_raw)
        
        if len(visual_desc.strip()) < 50:
            print(f"[gap_fill] WARNING: Short caption for frame at {timestamp_s:.1f}s")
            return None
        
        # Create Moment object (matching the schema used by AnalysisResult)
        return Moment(
            moment_id=f"moment_gap_{moment_index:03d}",
            beat_type="climax",  # Default to climax since these are action moments
            timestamp_range=(timestamp_s, timestamp_s + 5),  # 5-second window
            visual_description=visual_desc,
            key_dialogue=[],
            screenshot_candidates_s=[
                max(0, timestamp_s - 1),
                timestamp_s,
                timestamp_s + 1,
            ],
            emotional_beat=parsed.get("emotional_beat", "neutral"),
        )
        
    except Exception as e:
        print(f"[gap_fill] Failed to retrieve frame at {timestamp_s:.1f}s: {e}")
        return None


def fill_moment_gaps(
    story_text: str,
    analysis: AnalysisResult,
    subtitles: list[SubtitleSegment],
    video_path: Path,
    cache_dir: Path,
    franchise_db: Optional["FranchiseData"] = None,
    max_retrievals: int = 10,
    max_gap_percent: float = 0.30,
) -> tuple[AnalysisResult, GapFillResult]:
    """
    Main gap-fill function following VideoAgent philosophy.
    
    1. Extract actions from story
    2. Find gaps (actions without matching moments)
    3. Retrieve missing frames using semantic search + subtitle hints
    4. Return enriched analysis
    
    Args:
        story_text: The written story
        analysis: Current analysis result
        subtitles: Subtitle segments
        video_path: Path to video file
        cache_dir: Cache directory for frame index
        franchise_db: Optional franchise data
        max_retrievals: Maximum frames to retrieve (cap)
        max_gap_percent: If gaps exceed this percent of actions, warn
        
    Returns:
        Tuple of (enriched AnalysisResult, GapFillResult stats)
    """
    print(f"[gap_fill] Starting moment gap analysis...")
    
    # Step 1: Extract actions from story
    actions = extract_actions_from_story(story_text, franchise_db)
    print(f"[gap_fill] Extracted {len(actions)} actions from story")
    
    if not actions:
        return analysis, GapFillResult(
            actions_extracted=0,
            gaps_found=0,
            gaps_filled=0,
            gaps_unfillable=0,
            new_moments=[],
        )
    
    # Step 2: Find gaps (actions without matching moments)
    gaps: list[StoryAction] = []
    for action in actions:
        match = find_matching_moment(action, analysis, subtitles)
        if match is None:
            gaps.append(action)
    
    print(f"[gap_fill] Found {len(gaps)} gaps (actions without matching moments)")
    
    # Check if too many gaps (suggests upstream problem)
    gap_percent = len(gaps) / len(actions) if actions else 0
    if gap_percent > max_gap_percent:
        print(f"[gap_fill] WARNING: High gap rate ({gap_percent:.0%}). Video analysis may need improvement.")
    
    if not gaps:
        return analysis, GapFillResult(
            actions_extracted=len(actions),
            gaps_found=0,
            gaps_filled=0,
            gaps_unfillable=0,
            new_moments=[],
        )
    
    # Step 3: Try to load frame index for semantic search
    frame_index = load_frame_index(video_path, cache_dir)
    if frame_index:
        print(f"[gap_fill] Loaded frame index: {frame_index.n_frames} frames")
    else:
        print(f"[gap_fill] No frame index available, using subtitle hints only")
    
    # Step 4: Fill gaps (up to max_retrievals)
    new_moments: list[Moment] = []
    gaps_unfillable = 0
    gap_moment_index = 1
    
    for gap in gaps[:max_retrievals]:
        print(f"[gap_fill] Attempting to fill: {gap.action_phrase[:50]}...")
        
        timestamp = None
        
        # Primary: Semantic search on frame index
        if frame_index:
            results = semantic_search_frames(gap.action_phrase, frame_index, top_k=5)
            
            # Filter to candidates not too close to existing moments
            existing_timestamps = {m.timestamp_range[0] for m in analysis.moments}
            for result in results:
                if not any(abs(result.timestamp_s - ts) < 3.0 for ts in existing_timestamps):
                    timestamp = result.timestamp_s
                    print(f"[gap_fill]   Semantic search found: {timestamp:.1f}s (score: {result.similarity_score:.2f})")
                    break
        
        # Fallback: Subtitle timestamp hint
        if timestamp is None:
            subtitle_ts = find_timestamp_from_subtitles(gap, subtitles)
            if subtitle_ts:
                timestamp = subtitle_ts
                print(f"[gap_fill]   Subtitle hint found: {timestamp:.1f}s")
        
        # If we found a timestamp, retrieve and caption the frame
        if timestamp is not None:
            moment = retrieve_and_caption_frame(
                timestamp_s=timestamp,
                video_path=video_path,
                action=gap,
                moment_index=gap_moment_index,
                franchise_db=franchise_db,
            )
            if moment:
                new_moments.append(moment)
                gap_moment_index += 1
                print(f"[gap_fill]   ✓ Filled gap at {timestamp:.1f}s")
            else:
                gaps_unfillable += 1
                print(f"[gap_fill]   ✗ Failed to caption frame")
        else:
            gaps_unfillable += 1
            print(f"[gap_fill]   ✗ No suitable frame found")
    
    # Step 5: Merge new moments into analysis
    if new_moments:
        # Combine and sort by timestamp
        all_moments = list(analysis.moments) + new_moments
        all_moments.sort(key=lambda m: m.timestamp_range[0])
        
        # Create updated analysis
        analysis = AnalysisResult(
            title_candidates=analysis.title_candidates,
            characters=analysis.characters,
            beats=analysis.beats,
            moments=all_moments,
        )
        
        print(f"[gap_fill] Added {len(new_moments)} new moments (total: {len(all_moments)})")
    
    result = GapFillResult(
        actions_extracted=len(actions),
        gaps_found=len(gaps),
        gaps_filled=len(new_moments),
        gaps_unfillable=gaps_unfillable,
        new_moments=new_moments,
    )
    
    return analysis, result
