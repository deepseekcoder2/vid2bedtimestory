"""
Embedding module for semantic frame search.

Uses Qwen3-VL-Embedding-8B to create vector embeddings of video frames,
enabling semantic search across the entire video regardless of timestamps.

Components:
- FrameIndex: Pre-computed embeddings of all video frames
- semantic_search_frames(): Find frames matching a text query
- Reranker: Refine search results with Qwen3-VL-Reranker-8B

Usage:
    from vid2bedtimestory.embedding import FrameIndex, semantic_search_frames
    
    # Build index during video analysis
    index = FrameIndex.build(video_path, frame_paths)
    index.save(cache_path)
    
    # Search for frames matching a visual target
    results = semantic_search_frames(
        query="GT-Scorcher flying through the air with blue flames",
        index=index,
        top_k=30,
    )
    
Requirements:
    pip install torch transformers numpy
    
    For full functionality:
    pip install qwen-vl-utils
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Only import if numpy is available (core dependency)
if NUMPY_AVAILABLE:
    from .index import FrameIndex, build_frame_index, load_frame_index
    from .search import semantic_search_frames, SearchResult
    from .reranker import rerank_frames
    EMBEDDING_AVAILABLE = True
else:
    # Stub classes for when numpy isn't available
    FrameIndex = None
    build_frame_index = None
    load_frame_index = None
    semantic_search_frames = None
    SearchResult = None
    rerank_frames = None
    EMBEDDING_AVAILABLE = False

__all__ = [
    "FrameIndex",
    "build_frame_index", 
    "load_frame_index",
    "semantic_search_frames",
    "SearchResult",
    "rerank_frames",
    "EMBEDDING_AVAILABLE",
]
