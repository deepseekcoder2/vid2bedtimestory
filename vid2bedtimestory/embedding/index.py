"""
Frame embedding index for semantic video search.

Uses Qwen3-VL-Embedding-8B to create dense vector embeddings of video frames.
The index is cached per video and enables fast semantic search across all frames.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FrameEmbedding:
    """A single frame's embedding with metadata."""
    timestamp_s: float
    frame_path: Path
    embedding: np.ndarray  # Shape: (embedding_dim,)


class FrameIndex:
    """
    Index of frame embeddings for semantic search.
    
    Stores embeddings in a simple numpy format for fast cosine similarity search.
    """
    
    def __init__(
        self,
        video_hash: str,
        embeddings: np.ndarray,  # Shape: (n_frames, embedding_dim)
        timestamps: list[float],
        frame_paths: list[str],
        embedding_dim: int = 4096,  # Qwen3-VL-Embedding-8B default
    ):
        self.video_hash = video_hash
        self.embeddings = embeddings
        self.timestamps = timestamps
        self.frame_paths = frame_paths
        self.embedding_dim = embedding_dim
        
        # Pre-normalize embeddings for fast cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        self.embeddings_normalized = embeddings / norms
    
    @property
    def n_frames(self) -> int:
        return len(self.timestamps)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 30,
    ) -> list[tuple[float, float, str]]:
        """
        Search for frames similar to query embedding.
        
        Args:
            query_embedding: Query vector (shape: (embedding_dim,))
            top_k: Number of results to return
            
        Returns:
            List of (score, timestamp_s, frame_path) tuples, sorted by score descending
        """
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = self.embeddings_normalized @ query_normalized
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                float(similarities[idx]),
                self.timestamps[idx],
                self.frame_paths[idx],
            ))
        
        return results
    
    def save(self, path: Path) -> None:
        """Save index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON
        metadata = {
            "video_hash": self.video_hash,
            "timestamps": self.timestamps,
            "frame_paths": self.frame_paths,
            "embedding_dim": self.embedding_dim,
            "n_frames": self.n_frames,
        }
        
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f)
        
        # Save embeddings as numpy file
        np.save(path.with_suffix(".npy"), self.embeddings)
        
        print(f"[embedding] Saved index: {self.n_frames} frames to {path}")
    
    @classmethod
    def load(cls, path: Path) -> Optional["FrameIndex"]:
        """Load index from disk."""
        json_path = path.with_suffix(".json")
        npy_path = path.with_suffix(".npy")
        
        if not json_path.exists() or not npy_path.exists():
            return None
        
        with open(json_path, "r") as f:
            metadata = json.load(f)
        
        embeddings = np.load(npy_path)
        
        return cls(
            video_hash=metadata["video_hash"],
            embeddings=embeddings,
            timestamps=metadata["timestamps"],
            frame_paths=metadata["frame_paths"],
            embedding_dim=metadata["embedding_dim"],
        )


def _compute_video_hash(video_path: Path) -> str:
    """Compute a hash of the video file for cache keying."""
    stat = video_path.stat()
    key = f"{video_path.name}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def build_frame_index(
    video_path: Path,
    frame_paths: list[Path],
    timestamps: list[float],
    cache_dir: Optional[Path] = None,
    batch_size: int = 16,
) -> FrameIndex:
    """
    Build a frame embedding index for a video.
    
    Uses Qwen3-VL-Embedding-8B via PyTorch to embed all frames.
    
    Args:
        video_path: Path to source video (for cache keying)
        frame_paths: List of extracted frame image paths
        timestamps: Corresponding timestamps for each frame
        cache_dir: Directory to cache the index (optional)
        batch_size: Number of frames to embed at once
        
    Returns:
        FrameIndex for semantic search
    """
    if len(frame_paths) != len(timestamps):
        raise ValueError(f"Mismatched frame_paths ({len(frame_paths)}) and timestamps ({len(timestamps)})")
    
    video_hash = _compute_video_hash(video_path)
    
    # Check cache
    if cache_dir:
        cache_path = cache_dir / f"frame_index_{video_hash}"
        cached = FrameIndex.load(cache_path)
        if cached and cached.n_frames == len(frame_paths):
            print(f"[embedding] Loaded cached index: {cached.n_frames} frames")
            return cached
    
    print(f"[embedding] Building index for {len(frame_paths)} frames...")
    start_time = time.time()
    
    # Use subprocess worker to run PyTorch embedding
    worker_path = Path(__file__).parent / "embedding_worker.py"
    
    # Process in batches
    all_embeddings = []
    
    for batch_start in range(0, len(frame_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(frame_paths))
        batch_paths = frame_paths[batch_start:batch_end]
        
        print(f"[embedding]   Processing frames {batch_start+1}-{batch_end}/{len(frame_paths)}...")
        
        input_data = json.dumps({
            "action": "embed_images",
            "image_paths": [str(p.absolute()) for p in batch_paths],
        })
        
        try:
            result = subprocess.run(
                [sys.executable, str(worker_path)],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min per batch
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Embedding worker failed: {result.stderr[:500]}")
            
            output = json.loads(result.stdout)
            if not output.get("success"):
                raise RuntimeError(f"Embedding failed: {output.get('error')}")
            
            batch_embeddings = np.array(output["embeddings"])
            all_embeddings.append(batch_embeddings)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Embedding batch timed out")
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    
    elapsed = time.time() - start_time
    print(f"[embedding] Embedded {len(frame_paths)} frames in {elapsed:.1f}s")
    
    index = FrameIndex(
        video_hash=video_hash,
        embeddings=embeddings,
        timestamps=timestamps,
        frame_paths=[str(p) for p in frame_paths],
        embedding_dim=embeddings.shape[1],
    )
    
    # Cache the index
    if cache_dir:
        index.save(cache_path)
    
    return index


def load_frame_index(video_path: Path, cache_dir: Path) -> Optional[FrameIndex]:
    """
    Load a cached frame index for a video.
    
    Args:
        video_path: Path to source video
        cache_dir: Directory where index is cached
        
    Returns:
        FrameIndex if cached, None otherwise
    """
    video_hash = _compute_video_hash(video_path)
    cache_path = cache_dir / f"frame_index_{video_hash}"
    return FrameIndex.load(cache_path)
