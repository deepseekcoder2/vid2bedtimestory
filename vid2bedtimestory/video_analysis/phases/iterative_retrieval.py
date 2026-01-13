"""
Phase 1.5: Iterative Retrieval (VideoAgent-Style)

After sparse survey, LLM evaluates if information is sufficient.
If not, it generates queries for missing info, we retrieve matching frames
using embeddings, caption them, and repeat until confident.

This implements the core VideoAgent iterative loop from:
https://arxiv.org/abs/2403.10517
"""

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from vid2bedtimestory.llm import call_with_json_mode, LLMError as BaseLLMError
from vid2bedtimestory.models import SubtitleSegment

from ..types import FrameCaption, VLMError, LLMError
from ..prompts import SPARSE_SURVEY_PROMPT, get_sparse_survey_prompt
from ..config import get_config
from ..frame_utils import extract_frame_cached
from ..vlm_client import caption_frames_batch
from .sparse_survey import format_captions_for_llm

if TYPE_CHECKING:
    from vid2bedtimestory.knowledge import FranchiseData


# Maximum number of retrieval iterations
MAX_ITERATIONS = 3
# Maximum frames to retrieve per iteration
FRAMES_PER_ITERATION = 6

# Module-level variable to store franchise_db for _caption_new_frames
_current_franchise_db = None


def iterative_retrieval(
    video_path: Path,
    duration_s: float,
    initial_captions: list[FrameCaption],
    subtitles: list[SubtitleSegment],
    frame_index: Optional["FrameIndex"] = None,
    franchise_db: "FranchiseData" = None,
    subtitle_context_limit: int = 50,
) -> list[FrameCaption]:
    """
    VideoAgent-style iterative retrieval to fill information gaps.
    
    After initial sparse survey:
    1. LLM evaluates if information is sufficient (confidence 1-3)
    2. If insufficient, LLM generates queries for missing visual info
    3. Use embeddings to retrieve frames matching queries
    4. Caption new frames, add to pool
    5. Repeat until confident or max iterations
    
    Args:
        video_path: Path to video file
        duration_s: Total video duration
        initial_captions: Captions from sparse survey
        subtitles: Parsed subtitles for context
        frame_index: Pre-built embedding index (optional, will build if needed)
        franchise_db: Franchise database for prompt injection
        
    Returns:
        Augmented list of FrameCaption including retrieved frames
    """
    global _current_franchise_db
    _current_franchise_db = franchise_db
    
    all_captions = list(initial_captions)
    
    # Build or use provided frame index
    if frame_index is None:
        frame_index = _build_frame_index(video_path, duration_s)
    
    if frame_index is None:
        print("[iterative_retrieval] No embedding index available, skipping iterative retrieval")
        return all_captions
    
    # Format subtitles for context (limit configurable via --subtitle-context)
    subtitle_text = "\n".join(
        f"[{s.start_ms/1000:.1f}s] {s.text}" 
        for s in subtitles[:subtitle_context_limit]
    )
    
    for iteration in range(MAX_ITERATIONS):
        print(f"[iterative_retrieval] Iteration {iteration + 1}/{MAX_ITERATIONS}")
        
        # Step 1: Evaluate current information
        confidence, missing_queries = _evaluate_and_query(
            captions=all_captions,
            subtitle_text=subtitle_text,
            duration_s=duration_s,
        )
        
        print(f"[iterative_retrieval]   Confidence: {confidence}/3")
        
        # If confident enough, stop
        if confidence >= 3:
            print(f"[iterative_retrieval]   Sufficient information gathered")
            break
        
        if not missing_queries:
            print(f"[iterative_retrieval]   No specific queries generated, stopping")
            break
        
        print(f"[iterative_retrieval]   Missing info queries: {missing_queries}")
        
        # Step 2: Retrieve frames for each query
        new_timestamps = set()
        existing_timestamps = {fc.timestamp_s for fc in all_captions}
        
        for query in missing_queries[:3]:  # Max 3 queries per iteration
            retrieved = _retrieve_frames_for_query(
                query=query,
                frame_index=frame_index,
                existing_timestamps=existing_timestamps,
                max_frames=2,
            )
            new_timestamps.update(retrieved)
        
        if not new_timestamps:
            print(f"[iterative_retrieval]   No new frames found, stopping")
            break
        
        # Step 3: Extract and caption new frames
        new_captions = _caption_new_frames(
            video_path=video_path,
            timestamps=sorted(new_timestamps),
        )
        
        print(f"[iterative_retrieval]   Added {len(new_captions)} new frame captions")
        all_captions.extend(new_captions)
        
        # Sort by timestamp
        all_captions.sort(key=lambda fc: fc.timestamp_s)
    
    return all_captions


def _evaluate_and_query(
    captions: list[FrameCaption],
    subtitle_text: str,
    duration_s: float,
) -> tuple[int, list[str]]:
    """
    LLM evaluates if current captions are sufficient and generates queries for gaps.
    
    Returns:
        (confidence_level, list_of_queries)
        confidence: 1=insufficient, 2=partial, 3=sufficient
    """
    captions_text = format_captions_for_llm(captions)
    
    system_prompt = """You are analyzing a children's animated TV episode to understand its visual content.

Your task:
1. Evaluate if the current frame descriptions provide enough visual information
2. If NOT sufficient, generate specific queries for what's VISUALLY missing

Focus on:
- Key ACTION moments (not just dialogue scenes)
- Important OBJECTS that drive the plot
- Character INTERACTIONS and expressions
- Scene TRANSITIONS and locations

Do NOT ask for:
- More dialogue (we have subtitles for that)
- Abstract concepts (only visual elements)"""

    user_prompt = f"""Video duration: {duration_s:.1f} seconds

CURRENT FRAME DESCRIPTIONS:
{captions_text}

DIALOGUE CONTEXT:
{subtitle_text}

Evaluate the visual coverage:
1. Are there obvious GAPS in the timeline where important action might happen?
2. Are key OBJECTS mentioned in dialogue but not shown in any frame?
3. Are there CHARACTER ACTIONS that seem important but aren't captured?

Return JSON:
{{
    "confidence": 1/2/3,  // 1=major gaps, 2=some gaps, 3=sufficient
    "reasoning": "brief explanation",
    "missing_queries": [
        "specific visual query 1",
        "specific visual query 2"
    ]
}}

For missing_queries, be SPECIFIC and VISUAL:
- GOOD: "a spinning black tire on an orange track"
- GOOD: "close-up of character pressing a button"
- BAD: "what happens next" (too vague)
- BAD: "character's feelings" (not visual)"""

    try:
        response = call_with_json_mode(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=2,
        )
        
        confidence = int(response.get("confidence", 2))
        confidence = max(1, min(3, confidence))  # Clamp to 1-3
        
        queries = response.get("missing_queries", [])
        if not isinstance(queries, list):
            queries = []
        
        # Filter to only string queries
        queries = [q for q in queries if isinstance(q, str) and len(q) > 5]
        
        return confidence, queries
        
    except BaseLLMError as e:
        print(f"[iterative_retrieval] LLM evaluation failed: {e}")
        return 2, []  # Default to partial confidence, no queries


def _build_frame_index(video_path: Path, duration_s: float) -> Optional["FrameIndex"]:
    """
    Build embedding index for the video.
    
    Extracts frames at 1fps and embeds them for retrieval.
    """
    try:
        from vid2bedtimestory.embedding.index import FrameIndex, build_frame_index, load_frame_index
        from vid2bedtimestory.ffmpeg import extract_dense_frames
        import re
        
        # Check if we have a cached index
        cache_dir = Path("cache") / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try loading cached index first
        cached = load_frame_index(video_path, cache_dir)
        if cached:
            print(f"[iterative_retrieval] Loaded cached frame index: {cached.n_frames} frames")
            return cached
        
        # Extract frames at 1fps for embedding
        print(f"[iterative_retrieval] Building frame index (this may take a few minutes)...")
        frames_dir = cache_dir / f"{video_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = extract_dense_frames(
            video_path=video_path,
            out_dir=frames_dir,
            interval_s=1.0,  # 1 frame per second
        )
        
        if not frame_paths:
            return None
        
        # Extract timestamps from frame filenames (format: frame_NNNNNN.png where NNNNNN is ms)
        timestamps = []
        for fp in frame_paths:
            match = re.search(r'frame_(\d+)\.png', fp.name)
            if match:
                timestamps.append(float(match.group(1)) / 1000.0)  # ms to seconds
            else:
                # Fallback: infer from position
                idx = frame_paths.index(fp)
                timestamps.append(float(idx))
        
        # Build embedding index
        frame_index = build_frame_index(
            video_path=video_path,
            frame_paths=frame_paths,
            timestamps=timestamps,
            cache_dir=cache_dir,
        )
        
        print(f"[iterative_retrieval] Frame index built: {frame_index.n_frames} frames")
        
        return frame_index
        
    except ImportError as e:
        print(f"[iterative_retrieval] Embedding module not available: {e}")
        return None
    except Exception as e:
        print(f"[iterative_retrieval] Failed to build frame index: {e}")
        import traceback
        traceback.print_exc()
        return None


def _retrieve_frames_for_query(
    query: str,
    frame_index: "FrameIndex",
    existing_timestamps: set[float],
    max_frames: int = 2,
    use_reranker: bool = True,
) -> list[float]:
    """
    Use two-stage retrieval to find frames matching the query.
    
    Stage 1: Embedding search (fast) → Top 20 candidates
    Stage 2: Reranker (precise) → Score each, pick best
    
    Returns list of timestamps for frames that match and aren't already captured.
    """
    try:
        from vid2bedtimestory.embedding.search import semantic_search_frames
        
        # Stage 1: Fast embedding search for candidates
        candidates = semantic_search_frames(
            query=query,
            index=frame_index,
            top_k=20,  # Get 20 candidates for reranking
        )
        
        if not candidates:
            return []
        
        # Stage 2: Rerank for precision (if enabled)
        if use_reranker and len(candidates) > 1:
            try:
                from vid2bedtimestory.embedding.reranker import rerank_search_results
                from pathlib import Path
                
                reranked = rerank_search_results(
                    query=query,
                    search_results=candidates,
                    top_k=min(15, len(candidates)),  # Rerank top 15
                    instruction="Find the video frame that best matches this visual description. Reject blurry, dark, or transition frames.",
                )
                
                if reranked:
                    print(f"[iterative_retrieval]     Reranked: top score {reranked[0].relevance_score:.3f}")
                    # Use reranked results
                    candidates = [
                        type('SearchResult', (), {
                            'timestamp_s': r.timestamp_s,
                            'frame_path': r.frame_path,
                            'similarity_score': r.relevance_score,
                        })()
                        for r in reranked
                    ]
            except Exception as e:
                print(f"[iterative_retrieval]     Reranker failed, using embedding scores: {e}")
        
        # Filter out frames too close to existing ones (within 2 seconds)
        new_timestamps = []
        for result in candidates:
            ts = result.timestamp_s
            
            # Skip if too close to existing frame
            too_close = any(abs(ts - existing) < 2.0 for existing in existing_timestamps)
            if too_close:
                continue
            
            new_timestamps.append(ts)
            existing_timestamps.add(ts)  # Mark as used
            
            if len(new_timestamps) >= max_frames:
                break
        
        return new_timestamps
        
    except Exception as e:
        print(f"[iterative_retrieval] Retrieval failed for query '{query}': {e}")
        return []


def _caption_new_frames(
    video_path: Path,
    timestamps: list[float],
) -> list[FrameCaption]:
    """
    Extract and caption frames at given timestamps.
    """
    if not timestamps:
        return []
    
    # Extract frames
    frame_paths = []
    for ts in timestamps:
        try:
            frame_path = extract_frame_cached(video_path, ts)
            frame_paths.append((ts, frame_path))
        except Exception as e:
            print(f"[iterative_retrieval] Failed to extract frame at {ts:.1f}s: {e}")
            continue
    
    if not frame_paths:
        return []
    
    # Get franchise-specific prompt
    global _current_franchise_db
    if _current_franchise_db:
        prompt = get_sparse_survey_prompt(_current_franchise_db)
    else:
        prompt = SPARSE_SURVEY_PROMPT
    
    # Caption frames
    items = [(fp, prompt) for _, fp in frame_paths]
    
    try:
        captions = caption_frames_batch(items)
    except VLMError as e:
        print(f"[iterative_retrieval] VLM captioning failed: {e}")
        return []
    
    # Build FrameCaption objects
    results = []
    for (ts, frame_path), caption in zip(frame_paths, captions):
        results.append(FrameCaption(
            timestamp_s=ts,
            frame_path=frame_path,
            caption=caption,
        ))
    
    return results


# Type hint for optional import
try:
    from vid2bedtimestory.embedding.index import FrameIndex
except ImportError:
    FrameIndex = None
