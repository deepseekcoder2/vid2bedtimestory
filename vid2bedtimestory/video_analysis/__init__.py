"""
Video Analysis Module v2 (VideoAgent-Style)

This module implements phased video analysis inspired by the VideoAgent paper.
Instead of feeding all frames to a VLM at once, we:
1. Survey the video with sparse frames
2. Detect story beats using LLM on text
3. Deep-dive into each beat with targeted VLM calls
4. Extract character profiles with dedicated passes
5. Assemble all data into AnalysisResult

Usage:
    from vid2bedtimestory.video_analysis import analyze_video_v2
    
    result = analyze_video_v2(video_path, subtitles, duration_s)
    
    # With franchise database for known characters:
    from vid2bedtimestory.knowledge import load_franchise
    franchise_db = load_franchise("hot_wheels_lets_race")
    result = analyze_video_v2(video_path, subtitles, duration_s, franchise_db=franchise_db)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vid2bedtimestory.models import AnalysisResult, SubtitleSegment
from vid2bedtimestory.knowledge import FranchiseData

from .types import VideoAnalysisError, PhaseError
from .phases.sparse_survey import sparse_survey
from .phases.iterative_retrieval import iterative_retrieval
from .phases.beat_detection import detect_beats
from .phases.deep_dive import deep_dive
from .phases.character_extraction import extract_characters
from .assembly import assemble_analysis
from .refinement import refine_if_needed

__version__ = "2.0.0"
__all__ = ["analyze_video_v2", "VideoAnalysisError"]


def analyze_video_v2(
    video_path: Path,
    subtitles: list[SubtitleSegment],
    duration_s: float,
    skip_refinement: bool = False,
    franchise_db: FranchiseData = None,
    subtitle_context_limit: int = 50,
) -> AnalysisResult:
    """
    Analyze video using phased VideoAgent-style approach.
    
    This is the main entry point for video analysis. It runs the full pipeline:
    1. Sparse Survey - Sample 28 frames uniformly, caption each
    1.5. Iterative Retrieval - LLM evaluates gaps, embedding search retrieves missing frames
    2. Beat Detection - LLM identifies story structure from captions + subtitles
    3. Deep Dive - Dense captioning of 3-5 frames per story beat
    4. Character Extraction - Uses franchise DB or signal analysis + VLM discovery
    5. Assembly - Combine into final AnalysisResult
    6. Refinement - (Optional) Add frames if dialogue coverage is poor
    
    Args:
        video_path: Path to the video file (.mkv, .mp4, etc.)
        subtitles: List of parsed SubtitleSegment objects with timestamps
        duration_s: Total video duration in seconds
        skip_refinement: If True, skip the optional refinement pass
        franchise_db: REQUIRED franchise database for character/style info
    
    Returns:
        AnalysisResult with title_candidates, characters, beats, and moments
    
    Raises:
        VideoAnalysisError: If any phase fails
        ValueError: If franchise_db is not provided
    """
    if not franchise_db:
        raise ValueError("franchise_db is required for analyze_video_v2")
    franchise_name = franchise_db.franchise_name if franchise_db else "none"
    print(f"[video_analysis] Starting analysis of {video_path.name} ({duration_s:.1f}s)")
    print(f"[video_analysis] Franchise DB: {franchise_name}")
    
    # Phase 1: Sparse Survey
    print("[video_analysis] Phase 1: Sparse Survey...")
    try:
        sparse_captions = sparse_survey(video_path, duration_s, franchise_db=franchise_db)
        print(f"[video_analysis]   → {len(sparse_captions)} frames captioned")
    except Exception as e:
        raise PhaseError(f"Sparse survey failed: {e}", phase="sparse_survey") from e
    
    # Phase 1.5: Iterative Retrieval (VideoAgent-style gap filling)
    print("[video_analysis] Phase 1.5: Iterative Retrieval...")
    try:
        sparse_captions = iterative_retrieval(
            video_path=video_path,
            duration_s=duration_s,
            initial_captions=sparse_captions,
            subtitles=subtitles,
            franchise_db=franchise_db,
            subtitle_context_limit=subtitle_context_limit,
        )
        print(f"[video_analysis]   → {len(sparse_captions)} total frames after retrieval")
    except Exception as e:
        # Iterative retrieval failure is not critical - continue with initial captions
        print(f"[video_analysis]   → Iterative retrieval failed (non-critical): {e}")
    
    # Phase 2: Beat Detection
    print("[video_analysis] Phase 2: Beat Detection...")
    try:
        beats = detect_beats(sparse_captions, subtitles, duration_s, franchise_db=franchise_db)
        print(f"[video_analysis]   → {len(beats)} story beats identified")
    except Exception as e:
        raise PhaseError(f"Beat detection failed: {e}", phase="beat_detection") from e
    
    # Phase 3: Deep Dive
    print("[video_analysis] Phase 3: Deep Dive...")
    try:
        moments = deep_dive(
            video_path=video_path,
            beats=beats,
            subtitles=subtitles,
            franchise_db=franchise_db
        )
        print(f"[video_analysis]   → {len(moments)} moments captured")
    except Exception as e:
        raise PhaseError(f"Deep dive failed: {e}", phase="deep_dive") from e
    
    # Phase 4: Character Extraction
    print("[video_analysis] Phase 4: Character Extraction...")
    try:
        characters = extract_characters(
            video_path, 
            moments, 
            subtitles,
            franchise_db=franchise_db,
        )
        print(f"[video_analysis]   → {len(characters)} characters profiled")
    except Exception as e:
        raise PhaseError(
            f"Character extraction failed: {e}", 
            phase="character_extraction",
            partial_result={"sparse_captions": sparse_captions, "beats": beats, "moments": moments}
        ) from e
    
    # Phase 5: Assembly
    print("[video_analysis] Phase 5: Assembly...")
    try:
        analysis = assemble_analysis(
            sparse_captions=sparse_captions,
            beats=beats,
            moments=moments,
            characters=characters,
            duration_s=duration_s,
        )
        print(f"[video_analysis]   → AnalysisResult assembled")
    except Exception as e:
        raise PhaseError(
            f"Assembly failed: {e}",
            phase="assembly",
            partial_result={
                "sparse_captions": sparse_captions,
                "beats": beats,
                "moments": moments,
                "characters": characters,
            }
        ) from e
    
    # Phase 6: Refinement (Optional)
    if not skip_refinement:
        print("[video_analysis] Phase 6: Refinement check...")
        try:
            analysis = refine_if_needed(
                analysis=analysis,
                subtitles=subtitles,
                video_path=video_path,
                min_coverage=0.5,
            )
            print(f"[video_analysis]   → {len(analysis.moments)} moments after refinement")
        except Exception as e:
            # Refinement failure is not critical - return unrefined analysis
            print(f"[video_analysis]   → Refinement skipped due to error: {e}")
    
    print(f"[video_analysis] Analysis complete!")
    print(f"[video_analysis]   - {len(analysis.title_candidates)} title candidates")
    print(f"[video_analysis]   - {len(analysis.characters)} characters")
    print(f"[video_analysis]   - {len(analysis.beats)} beats")
    print(f"[video_analysis]   - {len(analysis.moments)} moments")
    
    return analysis
