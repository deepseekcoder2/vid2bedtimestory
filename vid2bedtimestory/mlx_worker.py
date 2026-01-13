#!/usr/bin/env python3
"""
Isolated MLX-VLM worker process.

Runs in a separate process to avoid Metal initialization conflicts.
"""

import sys
import json
from pathlib import Path


def run_analysis(video_path: str, subtitles_text: str, system_prompt: str, user_prompt: str, max_tokens: int = 256000) -> dict:
    """Run MLX-VLM analysis in isolated process."""
    # Import MLX FIRST before anything else
    from mlx_vlm import load, generate
    from mlx_vlm.video_generate import process_vision_info
    import mlx.core as mx

    # Load model - check HuggingFace cache first
    hf_cache_path = (
        Path.home()
        / ".cache/huggingface/hub/models--mlx-community--Qwen3-VL-32B-Instruct-8bit"
    )
    if hf_cache_path.exists():
        model_id = "mlx-community/Qwen3-VL-32B-Instruct-8bit"
    else:
        lm_studio_path = Path.home() / ".cache/lm-studio/models/lmstudio-community/Qwen3-VL-32B-Instruct-8bit"
        model_id = str(lm_studio_path) if lm_studio_path.exists() else "mlx-community/Qwen3-VL-32B-Instruct-8bit"

    model, processor = load(model_id)

    # Create messages
    # Qwen3-VL: 128K context, text-based time alignment for video
    # Settings optimized for 32GB+ RAM Macs
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 1280 * 720,  # 720p resolution
                "fps": 8.0,  # 8 fps for smooth action analysis (matches segment extraction)
            },
            {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
        ],
    }]

    # Process and generate
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs.get("pixel_values_videos", inputs.get("pixel_values")))
    attention_mask = mx.array(inputs["attention_mask"])

    kwargs = {}
    if inputs.get("video_grid_thw") is not None:
        kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])

    result = generate(model, processor, text, input_ids=input_ids, pixel_values=pixel_values,
                      mask=attention_mask, max_tokens=max_tokens, verbose=False, **kwargs)

    return {"text": result.text}


if __name__ == "__main__":
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())

    try:
        result = run_analysis(
            input_data["video_path"],
            input_data.get("subtitles_text", ""),
            input_data["system_prompt"],
            input_data["user_prompt"],
            input_data.get("max_tokens", 16000),
        )
        print(json.dumps({"success": True, "result": result}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
