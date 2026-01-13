"""
Semantic search over video frames.

Enables finding frames that match a text query semantically,
regardless of timestamp constraints.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .index import FrameIndex


@dataclass
class SearchResult:
    """A single search result."""
    timestamp_s: float
    frame_path: str
    similarity_score: float  # Cosine similarity (0-1)
    rank: int  # Position in results (1-indexed)


def semantic_search_frames(
    query: str,
    index: FrameIndex,
    top_k: int = 30,
) -> list[SearchResult]:
    """
    Search for frames matching a text query.
    
    Uses Qwen3-VL-Embedding-8B to embed the query and finds
    frames with highest cosine similarity.
    
    Args:
        query: Text description of desired frame content
        index: Pre-built frame embedding index
        top_k: Number of results to return
        
    Returns:
        List of SearchResult, sorted by similarity descending
    """
    # Get query embedding
    query_embedding = _embed_query(query)
    
    if query_embedding is None:
        print(f"[search] WARNING: Failed to embed query, returning empty results")
        return []
    
    # Search index
    raw_results = index.search(query_embedding, top_k=top_k)
    
    # Convert to SearchResult objects
    results = []
    for rank, (score, timestamp_s, frame_path) in enumerate(raw_results, start=1):
        results.append(SearchResult(
            timestamp_s=timestamp_s,
            frame_path=frame_path,
            similarity_score=score,
            rank=rank,
        ))
    
    return results


def _embed_query(query: str) -> Optional[np.ndarray]:
    """
    Embed a text query using Qwen3-VL-Embedding-8B.
    
    Runs in subprocess to isolate PyTorch from MLX.
    """
    worker_path = Path(__file__).parent / "embedding_worker.py"
    
    input_data = json.dumps({
        "action": "embed_text",
        "texts": [query],
    })
    
    try:
        result = subprocess.run(
            [sys.executable, str(worker_path)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            print(f"[search] Embedding worker failed: {result.stderr[:200]}")
            return None
        
        output = json.loads(result.stdout)
        if not output.get("success"):
            print(f"[search] Embedding failed: {output.get('error')}")
            return None
        
        return np.array(output["embeddings"][0])
        
    except subprocess.TimeoutExpired:
        print(f"[search] Query embedding timed out")
        return None
    except Exception as e:
        print(f"[search] Query embedding error: {e}")
        return None


def search_with_fallback(
    query: str,
    index: Optional[FrameIndex],
    fallback_timestamps: list[float],
    top_k: int = 30,
) -> list[SearchResult]:
    """
    Search frames with fallback to timestamp-based selection.
    
    If index is available, uses semantic search. Otherwise,
    returns results based on provided timestamps.
    
    Args:
        query: Text description of desired frame content
        index: Pre-built frame embedding index (optional)
        fallback_timestamps: Timestamps to use if index unavailable
        top_k: Number of results to return
        
    Returns:
        List of SearchResult
    """
    if index is not None:
        results = semantic_search_frames(query, index, top_k=top_k)
        if results:
            return results
    
    # Fallback: return results based on timestamps with fake scores
    results = []
    for rank, ts in enumerate(fallback_timestamps[:top_k], start=1):
        results.append(SearchResult(
            timestamp_s=ts,
            frame_path="",  # Will need to be extracted
            similarity_score=0.5,  # Neutral score
            rank=rank,
        ))
    
    return results
