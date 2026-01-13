"""
Frame reranking using Qwen3-VL-Reranker-8B.

Refines initial search results by computing precise relevance scores
between the text query and each candidate frame.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RerankResult:
    """A single reranked frame."""
    timestamp_s: float
    frame_path: str
    relevance_score: float  # Reranker score (0-1)
    original_rank: int  # Position before reranking
    new_rank: int  # Position after reranking


def rerank_frames(
    query: str,
    frame_paths: list[Path],
    timestamps: list[float],
    instruction: str = "Find the video frame that best matches this visual description. Pay attention to shot type requirements like [WIDE_ACTION] (full scene visible), [MEDIUM] (character with context), or [CLOSE_UP] (face/emotion focus).",
) -> list[RerankResult]:
    """
    Rerank candidate frames using Qwen3-VL-Reranker-8B.
    
    Takes initial candidates from embedding search and computes
    more precise relevance scores using cross-attention.
    
    Args:
        query: Visual target description
        frame_paths: List of candidate frame paths
        timestamps: Corresponding timestamps for each frame
        instruction: Task instruction for the reranker
        
    Returns:
        List of RerankResult, sorted by relevance_score descending
    """
    if len(frame_paths) != len(timestamps):
        raise ValueError("Mismatched frame_paths and timestamps")
    
    if not frame_paths:
        return []
    
    # Run reranker via subprocess
    worker_path = Path(__file__).parent / "reranker_worker.py"
    
    input_data = json.dumps({
        "instruction": instruction,
        "query": query,
        "image_paths": [str(p.absolute()) for p in frame_paths],
    })
    
    try:
        result = subprocess.run(
            [sys.executable, str(worker_path)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min for batch
        )
        
        if result.returncode != 0:
            print(f"[reranker] Worker failed: {result.stderr[:200]}")
            # Fallback: return original order with neutral scores
            return _fallback_results(frame_paths, timestamps)
        
        output = json.loads(result.stdout)
        if not output.get("success"):
            print(f"[reranker] Reranking failed: {output.get('error')}")
            return _fallback_results(frame_paths, timestamps)
        
        scores = output["scores"]
        
    except subprocess.TimeoutExpired:
        print(f"[reranker] Timed out")
        return _fallback_results(frame_paths, timestamps)
    except Exception as e:
        print(f"[reranker] Error: {e}")
        return _fallback_results(frame_paths, timestamps)
    
    # Build results with scores
    results = []
    for i, (path, ts, score) in enumerate(zip(frame_paths, timestamps, scores)):
        results.append(RerankResult(
            timestamp_s=ts,
            frame_path=str(path),
            relevance_score=score,
            original_rank=i + 1,
            new_rank=0,  # Will be set after sorting
        ))
    
    # Sort by score descending
    results.sort(key=lambda r: r.relevance_score, reverse=True)
    
    # Assign new ranks
    for i, r in enumerate(results):
        r.new_rank = i + 1
    
    return results


def _fallback_results(
    frame_paths: list[Path], 
    timestamps: list[float],
) -> list[RerankResult]:
    """Return results with neutral scores when reranker fails."""
    return [
        RerankResult(
            timestamp_s=ts,
            frame_path=str(path),
            relevance_score=0.5,
            original_rank=i + 1,
            new_rank=i + 1,
        )
        for i, (path, ts) in enumerate(zip(frame_paths, timestamps))
    ]


def rerank_search_results(
    query: str,
    search_results: list,  # List of SearchResult from search.py
    top_k: int = 10,
    instruction: str = "Find the image that best matches this visual description",
) -> list[RerankResult]:
    """
    Convenience function to rerank SearchResult objects.
    
    Args:
        query: Visual target description
        search_results: Results from semantic_search_frames()
        top_k: Number of results to rerank (for efficiency)
        instruction: Task instruction for the reranker
        
    Returns:
        Reranked results
    """
    # Take top candidates for reranking
    candidates = search_results[:top_k]
    
    if not candidates:
        return []
    
    frame_paths = [Path(r.frame_path) for r in candidates]
    timestamps = [r.timestamp_s for r in candidates]
    
    return rerank_frames(query, frame_paths, timestamps, instruction)
