#!/usr/bin/env python3
"""
Reranker worker subprocess for Qwen3-VL-Reranker-8B.

Runs in a separate process to isolate PyTorch from MLX and avoid Metal conflicts.
Uses the model with MPS support for Apple Silicon.

Input (JSON via stdin):
{
    "instruction": "Find the image that best matches this description",
    "query": "GT-Scorcher flying through the air with blue flames",
    "image_paths": ["path1.png", "path2.png", ...]
}

Output (JSON via stdout):
{
    "success": true,
    "scores": [0.92, 0.45, 0.78, ...]  # Relevance score per image
}
"""

import json
import sys
from pathlib import Path


def main():
    # Parse input
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}))
        sys.exit(1)
    
    instruction = request.get("instruction", "Find the video frame that best matches this visual description. Pay attention to shot type like [WIDE_ACTION] (full scene), [MEDIUM] (character with context), or [CLOSE_UP] (face focus).")
    query = request.get("query", "")
    image_paths = request.get("image_paths", [])
    
    if not query:
        print(json.dumps({"success": False, "error": "No query provided"}))
        sys.exit(1)
    
    if not image_paths:
        print(json.dumps({"success": False, "error": "No images provided"}))
        sys.exit(1)
    
    try:
        scores = rerank_images(instruction, query, image_paths)
        print(json.dumps({
            "success": True,
            "scores": scores,
        }))
    except Exception as e:
        import traceback
        print(json.dumps({
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }))
        sys.exit(1)


def get_device():
    """Get the best available device."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_path() -> str:
    """Get the model path. Returns HuggingFace model ID to auto-download."""
    # Check LM Studio cache first
    local_model_path = Path.home() / ".cache/lm-studio/models/qwen/Qwen3-VL-Reranker-2B"
    if local_model_path.exists():
        return str(local_model_path)
    
    # Fall back to HuggingFace - will auto-download to ~/.cache/huggingface/hub/
    return "Qwen/Qwen3-VL-Reranker-2B"


def rerank_images(instruction: str, query: str, image_paths: list[str]) -> list[float]:
    """
    Rerank images using Qwen3-VL-Reranker-8B.
    
    Returns relevance scores (0-1) for each image.
    """
    import torch
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    model_path = get_model_path()
    device = get_device()
    
    print(f"# Using device: {device}", file=sys.stderr)
    print(f"# Using model: {model_path}", file=sys.stderr)
    
    # Load model
    print(f"# Loading model weights...", file=sys.stderr)
    lm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    lm = lm.to(device)
    
    model = lm.model  # The base model without LM head
    model.eval()
    
    print(f"# Loading processor...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    # Create binary scoring layer from yes/no tokens
    token_true_id = processor.tokenizer.get_vocab()["yes"]
    token_false_id = processor.tokenizer.get_vocab()["no"]
    
    lm_head_weights = lm.lm_head.weight.data
    weight_yes = lm_head_weights[token_true_id]
    weight_no = lm_head_weights[token_false_id]
    
    D = weight_yes.size()[0]
    score_linear = torch.nn.Linear(D, 1, bias=False)
    with torch.no_grad():
        score_linear.weight[0] = weight_yes - weight_no
    score_linear = score_linear.to(device).to(model.dtype)
    score_linear.eval()
    
    scores = []
    
    for i, path in enumerate(image_paths):
        print(f"# Scoring image {i+1}/{len(image_paths)}: {Path(path).name}", file=sys.stderr)
        
        # Load image
        image = Image.open(path).convert("RGB")
        
        # Build reranker format
        conversation = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                }]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<Instruct>: {instruction}"},
                    {"type": "text", "text": "<Query>:"},
                    {"type": "text", "text": query},
                    {"type": "text", "text": "\n<Document>:"},
                    {"type": "image", "image": image},
                ]
            }
        ]
        
        # Process
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=text,
            images=[image],
            return_tensors='pt',
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Compute score
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state[:, -1]
            score = score_linear(last_hidden)
            score = torch.sigmoid(score).squeeze(-1).cpu().item()
        
        scores.append(score)
    
    return scores


if __name__ == "__main__":
    main()
