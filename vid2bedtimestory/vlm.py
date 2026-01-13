from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import requests

from . import prompts
from .config import config


class VLMError(RuntimeError):
    pass


# =============================================================================
# CLOUD VLM SCORING (OpenRouter)
# =============================================================================

def _load_image_base64(image_path: Path) -> str:
    """Load image and encode as base64 data URL."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    # Determine mime type
    suffix = image_path.suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_types.get(suffix, "image/png")
    
    return f"data:{mime};base64,{data}"


def _score_frame_cloud_single(
    image_path: Path,
    visual_target: str,
    model: str,
    api_key: str,
    base_url: str,
    max_retries: int = 3,
    scoring_guidance: str = "",
) -> Tuple[float, str]:
    """
    Score a single frame using OpenRouter vision API.
    
    Includes retry with exponential backoff for rate limits (429).
    
    Returns:
        (score, raw_text) tuple
    """
    import time
    
    prompt = prompts.VLM_FRAME_SCORE_PROMPT.format(visual_target=visual_target)
    if scoring_guidance:
        prompt = prompt + "\n\n" + scoring_guidance
    
    # Load image as base64
    image_data_url = _load_image_base64(image_path)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vid2bedtimestory/vid2bedtimestory",
        "X-Title": "Vid2BedtimeStory Frame Scoring",
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 32,
        "temperature": 0.1,
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            
            # Handle rate limiting with retry
            if response.status_code == 429:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            
            result = response.json()
            raw = result["choices"][0]["message"]["content"].strip()
            
            # Parse score from response
            token = raw.split()[0] if raw else ""
            token = token.strip().strip('"').strip("'").rstrip(".")
            
            try:
                score = float(token)
                score = max(1.0, min(10.0, score))
            except ValueError:
                score = 1.0
            
            return score, raw
            
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return 1.0, f"error: {e}"
        except (KeyError, IndexError) as e:
            return 1.0, f"error: invalid response format: {e}"
    
    return 1.0, f"error: max retries exceeded, last error: {last_error}"


def score_frames_cloud(
    frame_paths: list[Path],
    visual_target: str,
    model: str = None,
    max_concurrent: int = 10,
    scoring_guidance: str = "",
) -> list[Tuple[float, str]]:
    """
    Score multiple frames using OpenRouter vision API with parallel requests.
    
    Much faster than local VLM - parallelizes across API calls.
    
    Args:
        frame_paths: List of paths to frame images
        visual_target: The visual target to score against
        model: OpenRouter model ID (default from config)
        max_concurrent: Maximum concurrent API requests
        scoring_guidance: Optional franchise-specific scoring guidance
    
    Returns:
        List of (score, raw_text) tuples, one per frame.
    """
    if not frame_paths:
        return []
    
    # Load API key
    from .llm import load_api_key
    
    model = model or config.llm.vlm_cloud_model
    base_url = config.llm.creative_base_url
    api_key = load_api_key(base_url)
    
    print(f"[vlm_cloud] Scoring {len(frame_paths)} frames with {model} (max {max_concurrent} concurrent)")
    
    # Use ThreadPoolExecutor for parallel HTTP requests
    results: list[Tuple[float, str]] = [None] * len(frame_paths)
    
    def score_one(idx: int, path: Path) -> Tuple[int, Tuple[float, str]]:
        result = _score_frame_cloud_single(path, visual_target, model, api_key, base_url, scoring_guidance=scoring_guidance)
        return idx, result
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(score_one, i, path)
            for i, path in enumerate(frame_paths)
        ]
        
        completed = 0
        for future in futures:
            idx, result = future.result()
            results[idx] = result
            completed += 1
            
            # Progress indicator
            if completed % 10 == 0 or completed == len(frame_paths):
                print(f"[vlm_cloud]   {completed}/{len(frame_paths)} frames scored")
    
    return results


def score_frame_image(image_path: Path, visual_target: str) -> Tuple[float, str]:
    """
    Score a frame image against a visual target using MLX-VLM (Qwen3-VL) locally.

    Returns:
        (score, raw_text)
    """
    worker_path = Path(__file__).parent / "mlx_image_worker.py"
    if not worker_path.exists():
        raise VLMError(f"MLX image worker script not found: {worker_path}")

    prompt = prompts.VLM_FRAME_SCORE_PROMPT.format(visual_target=visual_target)

    input_data = json.dumps(
        {
            "image_path": str(image_path.absolute()),
            "prompt": prompt,
        }
    )

    try:
        result = subprocess.run(
            [sys.executable, str(worker_path)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise VLMError("VLM image scoring timed out after 5 minutes") from e

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "No error output"
        raise VLMError(f"MLX image worker failed (exit {result.returncode}):\n{stderr}")

    if not result.stdout.strip():
        raise VLMError("MLX image worker returned empty output")

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise VLMError(f"Failed to parse MLX image worker output as JSON: {e}") from e

    if not output.get("success"):
        raise VLMError(f"VLM image scoring failed: {output.get('error', 'Unknown error')}")

    raw = (output.get("result") or {}).get("text", "")
    raw_stripped = raw.strip()

    # Parse the first token as a float (model contract: returns only a number).
    token = raw_stripped.split()[0] if raw_stripped else ""
    token = token.strip().strip('"').strip("'")
    try:
        score = float(token)
    except ValueError as e:
        raise VLMError(f"VLM returned non-numeric score: {raw_stripped!r}") from e

    return score, raw_stripped


def score_frames_batch(
    frame_paths: list[Path],
    visual_target: str,
    timeout_s: int = 300,
    scoring_guidance: str = "",
) -> list[tuple[float, str]]:
    """
    Score multiple frames against a visual target using batch VLM processing.
    
    Loads the VLM model ONCE and scores all frames sequentially.
    Much faster than calling score_frame_image() individually.
    
    Args:
        frame_paths: List of paths to frame images
        visual_target: The visual target to score against
        timeout_s: Timeout for the entire batch (default 5 min)
        scoring_guidance: Optional franchise-specific scoring guidance
    
    Returns:
        List of (score, raw_text) tuples, one per frame.
        On individual frame errors, returns (1.0, "error: ...") for that frame.
    """
    if not frame_paths:
        return []
    
    worker_path = Path(__file__).parent / "mlx_batch_image_worker.py"
    if not worker_path.exists():
        raise VLMError(f"MLX batch image worker not found: {worker_path}")
    
    prompt = prompts.VLM_FRAME_SCORE_PROMPT.format(visual_target=visual_target)
    if scoring_guidance:
        prompt = prompt + "\n\n" + scoring_guidance
    
    # Prepare batch input
    images = [
        {"image_path": str(fp.absolute()), "prompt": prompt}
        for fp in frame_paths
    ]
    input_data = json.dumps({"images": images, "max_tokens": 32})
    
    try:
        result = subprocess.run(
            [sys.executable, str(worker_path)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise VLMError(f"VLM batch scoring timed out after {timeout_s}s") from e
    
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "No error output"
        raise VLMError(f"MLX batch worker failed (exit {result.returncode}):\n{stderr[:500]}")
    
    if not result.stdout.strip():
        raise VLMError("MLX batch worker returned empty output")
    
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise VLMError(f"Failed to parse batch worker output: {e}") from e
    
    if not output.get("success"):
        raise VLMError(f"VLM batch scoring failed: {output.get('error', 'Unknown')}")
    
    raw_results = output.get("results", [])
    
    # Parse scores from results
    parsed: list[tuple[float, str]] = []
    for res in raw_results:
        if "error" in res:
            parsed.append((1.0, f"error: {res['error']}"))
            continue
        
        raw = res.get("text", "").strip()
        token = raw.split()[0] if raw else ""
        token = token.strip().strip('"').strip("'")
        
        try:
            score = float(token)
            score = max(1.0, min(10.0, score))  # Clamp to 1-10
        except ValueError:
            score = 1.0  # Default on parse failure
        
        parsed.append((score, raw))
    
    return parsed
