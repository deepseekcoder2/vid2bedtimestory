#!/usr/bin/env python3
"""
Isolated MLX-VLM worker process for scoring a single image against a visual target.

Runs in a separate process to avoid Metal initialization conflicts with Rich/Typer.

Contract:
- stdin: JSON {"image_path": "...", "prompt": "..."}
- stdout: JSON {"success": true, "result": {"text": "..."} } OR {"success": false, "error": "..."}
"""

import json
import sys
from pathlib import Path


def _load_mlx_model():
    # Import MLX FIRST before anything else
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


def score_image(image_path: str, prompt: str) -> dict:
    # Import MLX FIRST before anything else
    from mlx_vlm import generate
    import mlx.core as mx

    # process_vision_info location varies across mlx-vlm versions; try both.
    try:
        from mlx_vlm.video_generate import process_vision_info  # type: ignore
    except Exception:  # pragma: no cover
        from mlx_vlm.utils import process_vision_info  # type: ignore

    model, processor = _load_mlx_model()

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
    # Some versions provide image_grid_thw; handle defensively.
    if inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])
    if inputs.get("video_grid_thw") is not None:
        kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])

    # Score generation - allow reasoning space even though output is a number
    result = generate(
        model,
        processor,
        text,
        input_ids=input_ids,
        pixel_values=pixel_values,
        mask=attention_mask,
        max_tokens=1024,
        verbose=False,
        **kwargs,
    )

    return {"text": result.text}


if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    try:
        result = score_image(
            input_data["image_path"],
            input_data["prompt"],
        )
        print(json.dumps({"success": True, "result": result}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


