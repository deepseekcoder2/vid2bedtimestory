#!/usr/bin/env python3
"""
Isolated MLX-VLM batch worker process for scoring MULTIPLE images.

Runs in a separate process to avoid Metal initialization conflicts.
Loads the model ONCE and processes all images sequentially.

Contract:
- stdin: JSON {"images": [{"image_path": "...", "prompt": "..."}, ...]}
- stdout: JSON {"success": true, "results": [{"text": "..."}, ...]} 
         OR {"success": false, "error": "..."}
"""

import json
import sys
from pathlib import Path


def _load_mlx_model():
    """Load MLX-VLM model ONCE for batch processing."""
    from mlx_vlm import load

    # Check HuggingFace cache first (preferred)
    hf_cache_path = (
        Path.home()
        / ".cache/huggingface/hub/models--mlx-community--Qwen3-VL-32B-Instruct-8bit"
    )
    if hf_cache_path.exists():
        return load("mlx-community/Qwen3-VL-32B-Instruct-8bit")
    
    # Check LM Studio cache
    lm_studio_path = (
        Path.home()
        / ".cache/lm-studio/models/lmstudio-community/Qwen3-VL-32B-Instruct-8bit"
    )
    if lm_studio_path.exists():
        return load(str(lm_studio_path))
    
    # Fallback - will attempt download
    return load("mlx-community/Qwen3-VL-32B-Instruct-8bit")


def score_single_image(model, processor, image_path: str, prompt: str, max_tokens: int = 1024) -> dict:
    """Score a single image using already-loaded model."""
    from mlx_vlm import generate
    import mlx.core as mx

    # process_vision_info location varies across mlx-vlm versions
    try:
        from mlx_vlm.video_generate import process_vision_info
    except Exception:
        from mlx_vlm.utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "max_pixels": 1280 * 720,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs.get("pixel_values", inputs.get("pixel_values_videos")))
    attention_mask = mx.array(inputs["attention_mask"])

    kwargs = {}
    if inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])
    if inputs.get("video_grid_thw") is not None:
        kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])

    result = generate(
        model,
        processor,
        text,
        input_ids=input_ids,
        pixel_values=pixel_values,
        mask=attention_mask,
        max_tokens=max_tokens,
        verbose=False,
        **kwargs,
    )

    # Handle different return types from generate()
    if hasattr(result, 'text'):
        return {"text": result.text}
    else:
        return {"text": str(result)}


def score_batch(images: list[dict], max_tokens: int = 1024) -> list[dict]:
    """
    Score multiple images with a single model load.
    
    Args:
        images: List of {"image_path": str, "prompt": str}
        max_tokens: Max tokens per response
    
    Returns:
        List of {"text": str} or {"error": str} for each image
    """
    # Load model ONCE
    model, processor = _load_mlx_model()
    
    results = []
    for i, item in enumerate(images):
        try:
            result = score_single_image(
                model, 
                processor,
                item["image_path"],
                item["prompt"],
                max_tokens=max_tokens,
            )
            results.append(result)
        except Exception as e:
            # Individual image failure doesn't stop the batch
            results.append({"error": str(e)})
    
    return results


if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    
    try:
        images = input_data.get("images", [])
        max_tokens = input_data.get("max_tokens", 1024)
        
        if not images:
            print(json.dumps({"success": False, "error": "No images provided"}))
            sys.exit(1)
        
        results = score_batch(images, max_tokens=max_tokens)
        print(json.dumps({"success": True, "results": results}))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

