#!/usr/bin/env python3
"""
Embedding worker subprocess for Qwen3-VL-Embedding-8B.

Runs in a separate process to isolate PyTorch from MLX and avoid Metal conflicts.
Directly uses the model with MPS support for Apple Silicon.

Input (JSON via stdin):
{
    "action": "embed_images" | "embed_text",
    "image_paths": ["path1.png", ...],  # for embed_images
    "texts": ["query text", ...],       # for embed_text
}

Output (JSON via stdout):
{
    "success": true,
    "embeddings": [[float, ...], ...]  # List of embedding vectors
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
    
    action = request.get("action")
    
    if action == "embed_images":
        return embed_images(request.get("image_paths", []))
    elif action == "embed_text":
        return embed_text(request.get("texts", []))
    else:
        print(json.dumps({"success": False, "error": f"Unknown action: {action}"}))
        sys.exit(1)


def get_model_path() -> str:
    """Get the model path. Returns HuggingFace model ID to auto-download."""
    # Check LM Studio cache first
    local_model_path = Path.home() / ".cache/lm-studio/models/qwen/Qwen3-VL-Embedding-2B"
    if local_model_path.exists():
        return str(local_model_path)
    
    # Fall back to HuggingFace - will auto-download to ~/.cache/huggingface/hub/
    return "Qwen/Qwen3-VL-Embedding-2B"


def get_device():
    """Get the best available device."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_embedder():
    """Load the Qwen3VL embedder with MPS support."""
    import torch
    import torch.nn.functional as F
    
    model_path = get_model_path()
    print(f"# Using model: {model_path}", file=sys.stderr)
    
    # Import the custom model class (using trust_remote_code)
    from transformers import AutoModel, AutoProcessor
    
    device = get_device()
    print(f"# Using device: {device}", file=sys.stderr)
    
    # Load model with explicit device placement
    print(f"# Loading model weights...", file=sys.stderr)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    
    print(f"# Loading processor...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    return model, processor, device


def pooling_last(hidden_state, attention_mask):
    """Pool the last hidden state by attention mask for embeddings."""
    import torch
    flipped_tensor = attention_mask.flip(dims=[1])
    last_one_positions = flipped_tensor.argmax(dim=1)
    col = attention_mask.shape[1] - last_one_positions - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


def embed_images(image_paths: list[str]):
    """Embed a batch of images using Qwen3-VL-Embedding-8B."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    
    try:
        model, processor, device = load_embedder()
        
        # Import vision processing utility
        model_path = get_model_path()
        scripts_path = model_path / "scripts"
        sys.path.insert(0, str(scripts_path))
        
        embeddings = []
        
        for i, path in enumerate(image_paths):
            print(f"# Embedding image {i+1}/{len(image_paths)}: {Path(path).name}", file=sys.stderr)
            
            # Load image
            image = Image.open(path).convert("RGB")
            
            # Build conversation format expected by the model
            # Use retrieval-specific instruction for building searchable index
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": "Represent this image for retrieval. Focus on the scene composition, characters, actions, and visual elements."}]},
                {"role": "user", "content": [{"type": "image", "image": image}]}
            ]
            
            # Process with the processor
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            
            inputs = processor(
                text=text,
                images=[image],
                return_tensors='pt',
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                emb = pooling_last(outputs.last_hidden_state, inputs['attention_mask'])
                emb = F.normalize(emb, p=2, dim=-1)
            
            embeddings.append(emb[0].cpu().float().numpy().tolist())
        
        print(json.dumps({
            "success": True,
            "embeddings": embeddings,
        }))
        
    except Exception as e:
        import traceback
        print(json.dumps({
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }))
        sys.exit(1)


def embed_text(texts: list[str]):
    """Embed text queries using Qwen3-VL-Embedding-8B."""
    import torch
    import torch.nn.functional as F
    
    try:
        model, processor, device = load_embedder()
        
        embeddings = []
        
        for text in texts:
            # Build conversation format
            # Use query instruction for searching the image index
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": "Represent this query for finding relevant images. The query describes a scene from a children's animated show."}]},
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ]
            
            # Process with the processor
            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            
            inputs = processor(
                text=prompt,
                return_tensors='pt',
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                emb = pooling_last(outputs.last_hidden_state, inputs['attention_mask'])
                emb = F.normalize(emb, p=2, dim=-1)
            
            embeddings.append(emb[0].cpu().float().numpy().tolist())
        
        print(json.dumps({
            "success": True,
            "embeddings": embeddings,
        }))
        
    except Exception as e:
        import traceback
        print(json.dumps({
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
