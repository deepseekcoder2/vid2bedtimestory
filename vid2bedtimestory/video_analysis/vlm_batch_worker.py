#!/usr/bin/env python3
"""
Batch VLM worker for video analysis.

This script runs as a subprocess to:
1. Isolate MLX/Metal initialization from the main process
2. Load the model ONCE and process multiple frames
3. Return results as JSON

Protocol:
- Input (stdin): JSON with model_path, items[], max_tokens
- Output (stdout): JSON with success, results[] or error

Usage:
    echo '{"model_path": "...", "items": [...], "max_tokens": 512}' | python vlm_batch_worker.py
"""

import json
import sys
from pathlib import Path


def main():
    """Main entry point for batch worker."""
    # Read input
    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}))
        return
    
    model_path = input_data.get("model_path")
    items = input_data.get("items", [])
    max_tokens = input_data.get("max_tokens", 16384)  # Generous for image captioning with reasoning
    
    if not items:
        print(json.dumps({"success": True, "results": []}))
        return
    
    try:
        # Import MLX FIRST before any other imports that might touch Metal
        from mlx_vlm import load, generate
        import mlx.core as mx
        
        # Import process_vision_info (location varies by mlx-vlm version)
        try:
            from mlx_vlm.video_generate import process_vision_info
        except ImportError:
            from mlx_vlm.utils import process_vision_info
        
        # Load model once
        model, processor = load(model_path)
        
        results = []
        for item in items:
            image_path = item["image_path"]
            prompt = item["prompt"]
            
            # Build message
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "max_pixels": 1280 * 720,  # 720p
                    },
                    {"type": "text", "text": prompt},
                ],
            }]
            
            # Process
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, video_inputs, _ = process_vision_info(
                messages, 
                return_video_kwargs=True
            )
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            input_ids = mx.array(inputs["input_ids"])
            pixel_values = mx.array(
                inputs.get("pixel_values", inputs.get("pixel_values_videos"))
            )
            attention_mask = mx.array(inputs["attention_mask"])
            
            # Handle optional kwargs
            kwargs = {}
            if inputs.get("image_grid_thw") is not None:
                kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])
            if inputs.get("video_grid_thw") is not None:
                kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
            
            # Generate
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
            
            if hasattr(result, 'text'):
                results.append({"text": result.text})
            else:
                results.append({"text": str(result)})
        
        print(json.dumps({"success": True, "results": results}))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()

