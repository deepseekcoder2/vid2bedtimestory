"""
VLM (Vision-Language Model) client for video analysis.

Provides single-image captioning with:
- Automatic caching of results
- Subprocess isolation to avoid Metal/GPU conflicts
- Batch processing for efficiency
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from typing import Optional, Any

from .cache import get_cached_caption, cache_caption
from .config import get_config
from .types import VLMError


def extract_json_robust(text: str) -> dict[str, Any]:
    """
    Extract JSON from potentially chatty model responses.
    
    Handles markdown blocks, conversational preamble, and common formatting errors.
    """
    # 1. Try simple parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Try to find the first '{' and last '}'
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Last resort: build a skeleton if we can find a visual_description key
    # This handles models that cut off the closing brace
    if '"visual_description":' in text:
        # Extract everything after visual_description":
        parts = text.split('"visual_description":', 1)
        if len(parts) > 1:
            desc = parts[1].strip().strip('"').strip("'").split('"', 1)[0]
            return {
                "visual_description": desc,
                "emotional_beat": "neutral"
            }

    # If all failed, return raw text in a skeleton
    return {
        "visual_description": text[:1000], # Clamp for safety
        "emotional_beat": "neutral"
    }


class VLMClient:
    """
    Client for VLM image captioning.
    
    Hybrid backend support:
    - Phase 1 (Survey): Local MLX-VLM (efficiency)
    - Phase 3 (Deep Dive): Cloud OpenRouter 235B (highest quality)
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """
        Initialize VLM client.
        
        Args:
            model_path: Path to MLX model (uses default if None)
            use_cache: Whether to use caption caching
        """
        self.model_path = model_path or self._default_model_path()
        self.use_cache = use_cache
        self._worker_path = Path(__file__).parent / "vlm_batch_worker.py"
    
    @staticmethod
    def _default_model_path() -> Path:
        """Get default model path - checks HuggingFace cache first."""
        # Check HuggingFace cache (preferred - where models are typically downloaded)
        hf_cache_path = (
            Path.home()
            / ".cache/huggingface/hub/models--mlx-community--Qwen3-VL-32B-Instruct-8bit"
        )
        if hf_cache_path.exists():
            # Return the model ID - transformers will find it in cache
            return Path("mlx-community/Qwen3-VL-32B-Instruct-8bit")
        
        # Check LM Studio cache (alternative location)
        lm_studio_path = (
            Path.home() 
            / ".cache/lm-studio/models/lmstudio-community/Qwen3-VL-32B-Instruct-8bit"
        )
        if lm_studio_path.exists():
            return lm_studio_path
        
        # Fallback to HuggingFace model ID (will attempt download)
        return Path("mlx-community/Qwen3-VL-32B-Instruct-8bit")
    
    def caption_frame(
        self,
        frame_path: Path,
        prompt: str,
        force: bool = False,
        prefer_cloud: bool | None = None,
    ) -> str:
        """
        Caption a single frame.
        
        Args:
            frame_path: Path to the frame image (PNG)
            prompt: The prompt for captioning
            force: If True, bypass cache
            prefer_cloud: True=cloud, False=local, None=use config.llm.vlm_backend
            
        Returns:
            Caption text from VLM
            
        Raises:
            VLMError: If VLM call fails
        """
        # Check cache first
        if self.use_cache and not force:
            cached = get_cached_caption(frame_path, prompt)
            if cached:
                return cached
        
        # Call VLM (Batch of 1)
        results = self.caption_frames_batch([(frame_path, prompt)], force=force, prefer_cloud=prefer_cloud)
        return results[0] if results else ""
    
    def caption_frames_batch(
        self,
        items: list[tuple[Path, str]],
        force: bool = False,
        prefer_cloud: bool | None = None,
    ) -> list[str]:
        """
        Caption multiple frames efficiently.
        
        Args:
            items: List of (frame_path, prompt) tuples
            force: If True, bypass cache for all items
            prefer_cloud: True=cloud, False=local, None=use config.llm.vlm_backend
            
        Returns:
            List of caption strings (same order as input)
            
        Raises:
            VLMError: If VLM call fails
        """
        if not items:
            return []
        
        # Check cache for each item
        results: list[Optional[str]] = [None] * len(items)
        uncached_indices: list[int] = []
        
        if self.use_cache and not force:
            for i, (frame_path, prompt) in enumerate(items):
                cached = get_cached_caption(frame_path, prompt)
                if cached:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(len(items)))
        
        # All cached - return early
        if not uncached_indices:
            return [r for r in results if r is not None]
        
        # Batch call for uncached items
        batch_items = [items[i] for i in uncached_indices]
        
        from vid2bedtimestory.config import config as main_config
        # Video analysis (sparse_survey, deep_dive) should use LOCAL VLM by default
        # because it processes hundreds of frames - cloud would be expensive/slow.
        # Only use cloud if EXPLICITLY requested (prefer_cloud=True).
        # The config.llm.vlm_backend setting is for FRAME SCORING only.
        is_cloud = prefer_cloud is True
        
        if is_cloud:
            batch_results = self._call_vlm_cloud_batch(batch_items)
        else:
            batch_results = self._call_vlm_local_batch(batch_items)
        
        # Merge results and cache
        for idx, result in zip(uncached_indices, batch_results):
            results[idx] = result
            if self.use_cache:
                frame_path, prompt = items[idx]
                cache_caption(frame_path, prompt, result)
        
        return [r for r in results if r is not None]
    
    def _call_vlm_local_batch(self, items: list[tuple[Path, str]]) -> list[str]:
        """Execute local batch VLM call in subprocess (MLX)."""
        config = get_config()
        
        if not self._worker_path.exists():
            raise VLMError(f"VLM batch worker not found: {self._worker_path}")
        
        input_data = json.dumps({
            "model_path": str(self.model_path),
            "items": [
                {"image_path": str(fp.absolute()), "prompt": p}
                for fp, p in items
            ],
            "max_tokens": config.vlm_max_tokens,
        })
        
        # Calculate timeout based on number of items
        timeout = config.vlm_timeout_s * len(items)
        
        try:
            result = subprocess.run(
                [sys.executable, str(self._worker_path)],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise VLMError(
                f"VLM batch processing timed out after {timeout}s "
                f"for {len(items)} frames"
            ) from e
        
        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "No error output"
            raise VLMError(f"VLM batch worker failed (exit {result.returncode}):\n{stderr}")
        
        if not result.stdout.strip():
            raise VLMError("VLM batch worker returned empty output")
        
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise VLMError(f"Failed to parse VLM output as JSON: {e}") from e
        
        if not output.get("success"):
            raise VLMError(f"VLM batch processing failed: {output.get('error', 'Unknown')}")
        
        return [item["text"] for item in output["results"]]

    def _call_vlm_cloud_batch(self, items: list[tuple[Path, str]]) -> list[str]:
        """Execute parallel cloud VLM calls via OpenRouter 235B."""
        from vid2bedtimestory.config import config as main_config
        from vid2bedtimestory.llm import load_api_key
        import requests
        from concurrent.futures import ThreadPoolExecutor
        import base64
        import time

        model = main_config.llm.vlm_cloud_model
        base_url = main_config.llm.creative_base_url
        api_key = load_api_key(base_url)
        max_tokens = get_config().vlm_max_tokens

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vid2bedtimestory/vid2bedtimestory",
            "X-Title": "Vid2BedtimeStory Video Analysis",
        }

        def _call_one(item):
            frame_path, prompt = item
            try:
                with open(frame_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                }

                # Simple retry for 429
                for attempt in range(3):
                    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
                    if response.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    response.raise_for_status()
                    break
                
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[vlm_cloud] Error on {frame_path.name}: {e}")
                return f"Error: {e}"

        max_workers = main_config.llm.vlm_cloud_max_concurrent
        print(f"[vlm_cloud] Deep Diving {len(items)} frames with {model} ({max_workers} concurrent)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_call_one, items))


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

# Global client instance (lazy initialization)
_client: Optional[VLMClient] = None


def get_vlm_client() -> VLMClient:
    """Get the global VLM client instance."""
    global _client
    if _client is None:
        _client = VLMClient()
    return _client


def caption_frame(
    frame_path: Path,
    prompt: str,
    force: bool = False,
    prefer_cloud: bool | None = None,
) -> str:
    """
    Caption a single frame (convenience function).
    
    Uses the global VLM client instance.
    prefer_cloud: True=cloud, False=local, None=use config.llm.vlm_backend
    """
    return get_vlm_client().caption_frame(frame_path, prompt, force=force, prefer_cloud=prefer_cloud)


def caption_frames_batch(
    items: list[tuple[Path, str]],
    force: bool = False,
    prefer_cloud: bool | None = None,
) -> list[str]:
    """
    Caption multiple frames (convenience function).
    
    Uses the global VLM client instance.
    prefer_cloud: True=cloud, False=local, None=use config.llm.vlm_backend
    """
    return get_vlm_client().caption_frames_batch(items, force=force, prefer_cloud=prefer_cloud)

