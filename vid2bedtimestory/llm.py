from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import requests
from pydantic import BaseModel, ValidationError

from .config import config
from .models import AnalysisResult, BookSpec, PageSpec, VisualTarget, VisualTargetsResult
from . import prompts
from .ffmpeg import ffprobe_duration_s

if TYPE_CHECKING:
    from .knowledge.loader import FranchiseData


class LLMConfig(BaseModel):
    """Configuration for LLM calls (unused - see config.py for actual config)."""
    api_key: str
    model: str = "xiaomi/mimo-v2-flash:free"
    base_url: str = "https://openrouter.ai/api/v1"
    max_retries: int = 3
    timeout: int = 300


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


def load_api_key(base_url: str = None) -> str:
    """Load API key from the project file. Returns placeholder for local servers."""
    # For local servers (LM Studio, Ollama), no API key needed
    if base_url and ('localhost' in base_url or '192.168.' in base_url or '127.0.0.1' in base_url or '10.' in base_url):
        return "not-needed"
    
    api_key_file = Path(__file__).parent.parent / "openrouterapikey.md"
    try:
        with open(api_key_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            # Extract the key (first line, remove any markdown formatting)
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
        raise LLMError("No valid API key found in openrouterapikey.md")
    except FileNotFoundError:
        raise LLMError("openrouterapikey.md file not found")


def _write_llm_error_log(
    model: str,
    max_tokens: int,
    messages: list[Dict[str, Any]],
    empty_responses: list[dict],
) -> Path:
    """
    Write a diagnostic log file for LLM errors.
    
    Args:
        model: The model that was called
        max_tokens: Token limit used
        messages: The messages sent to the API
        empty_responses: List of empty response details from each attempt
        
    Returns:
        Path to the created log file
    """
    from datetime import datetime
    
    # Create logs directory in project root
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"llm_error_{timestamp}.log"
    
    # Build log content
    lines = [
        "=" * 60,
        "LLM ERROR DIAGNOSTIC LOG",
        "=" * 60,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Model: {model}",
        f"Max Tokens: {max_tokens}",
        f"Attempts: {len(empty_responses)}",
        "",
        "=" * 60,
        "PROMPT",
        "=" * 60,
    ]
    
    # Add messages
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            lines.append(f"[{role}]")
            lines.append(content)
            lines.append("")
        else:
            lines.append(f"[{role}] (non-string content: {type(content).__name__})")
            lines.append("")
    
    lines.extend([
        "=" * 60,
        "RESPONSES",
        "=" * 60,
    ])
    
    for resp in empty_responses:
        lines.append(f"Attempt {resp['attempt']}: \"{resp['content']}\" (finish_reason: {resp['finish_reason']})")
    
    lines.extend([
        "",
        "=" * 60,
        "TROUBLESHOOTING",
        "=" * 60,
        "1. Check if the model is available on OpenRouter",
        "2. Check your API key and rate limits",
        "3. Try running the same prompt manually",
        "4. Report this issue at: https://github.com/your-repo/bookify/issues",
        "   Please include this log file.",
    ])
    
    # Write to file
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return log_path


def call_llm(
    messages: list[Dict[str, Any]],
    response_format: Dict[str, str] = None,
    model: str = None,
    base_url: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """
    Call LLM API via OpenRouter (MiMo-V2-Flash).

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        response_format: Response format specification (unused for most models)
        model: Model to use (defaults to config.llm.creative_model)
        base_url: API base URL (defaults to config.llm.creative_base_url)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Raw response text from the LLM
    """
    model = model or config.llm.creative_model
    base_url = base_url or config.llm.creative_base_url
    api_key = load_api_key(base_url)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vid2bedtimestory/vid2bedtimestory",
        "X-Title": "Vid2BedtimeStory Video-to-Book Converter",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Force JSON mode if requested (only works for some models)
    if response_format:
        payload["response_format"] = response_format

    endpoint = f"{base_url}/chat/completions"
    
    # Log request details for debugging
    total_chars = sum(len(m.get("content", "")) if isinstance(m.get("content"), str) else 0 for m in messages)
    print(f"[llm] Calling {model} (max_tokens={max_tokens}, prompt_chars={total_chars})")
    
    # Retry logic for transient network errors and empty responses
    max_retries = 3
    last_error = None
    empty_responses = []  # Track empty responses for diagnostic log
    used_fallback = False  # Track if fallback model is used
    
    for attempt in range(max_retries):
        current_model = model
        if used_fallback:
            current_model = config.llm.utility_model_fallback
            print(f"[llm] Using fallback model: {current_model}")
        
        payload["model"] = current_model
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=config.llm.timeout if hasattr(config, 'llm') else 600,
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if response has expected structure
            if "choices" not in result:
                raise LLMError(f"Unexpected response format: {json.dumps(result, indent=2)}")
            
            content = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0].get("finish_reason", "unknown")
            
            # Check for content filter and switch to fallback model
            if finish_reason == "content_filter" and not used_fallback:
                print(f"[llm] WARNING: Content filter triggered for {model}. Switching to fallback model.")
                used_fallback = True
                empty_responses = []
                continue
            
            # Check for empty/None content and retry
            if content is None or content.strip() == "":
                empty_responses.append({
                    "attempt": attempt + 1,
                    "content": content,
                    "finish_reason": finish_reason,
                })
                print(f"[llm] WARNING: Empty response (attempt {attempt + 1}/{max_retries}, finish_reason: {finish_reason})")
                
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    print(f"[llm] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # All retries exhausted - write diagnostic log and raise error
                try:
                    log_path = _write_llm_error_log(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages,
                        empty_responses=empty_responses,
                    )
                    raise LLMError(
                        f"LLM returned empty response after {max_retries} attempts.\n"
                        f"Diagnostic log saved to: {log_path}\n"
                        f"Please include this file when reporting issues."
                    )
                except LLMError:
                    raise  # Re-raise the LLMError we just created
                except Exception as log_err:
                    # Log file write failed, still raise the main error
                    raise LLMError(
                        f"LLM returned empty response after {max_retries} attempts.\n"
                        f"(Failed to write diagnostic log: {log_err})"
                    )
            
            return content

        except requests.RequestException as e:
            last_error = e
            # Retry on transient network errors
            if attempt < max_retries - 1:
                import time
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"[llm] Network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            # Final attempt failed
            error_detail = ""
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = f"\nResponse: {e.response.text}"
            except:
                pass
            raise LLMError(f"OpenRouter API request failed after {max_retries} attempts: {e}{error_detail}")
        except (KeyError, IndexError) as e:
            raise LLMError(f"Invalid response format from OpenRouter: {e}")


def call_with_json_mode(
    system_prompt: str = None,
    user_prompt: str = None,
    messages: list[Dict[str, Any]] = None,
    expected_schema: type[BaseModel] = None,
    max_retries: int = 3,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    Call LLM with JSON mode enabled and validate response against Pydantic schema.

    Args:
        system_prompt: System instructions (used if messages not provided)
        user_prompt: User prompt (used if messages not provided)
        messages: Custom messages array (for multimodal content)
        expected_schema: Pydantic model to validate against
        max_retries: Number of retry attempts
        max_tokens: Maximum tokens to generate (increase for large JSON output)

    Returns:
        Parsed and validated JSON response
    """
    # Build messages if not provided
    if messages is None:
        if system_prompt is None or user_prompt is None:
            raise ValueError("Either messages or both system_prompt and user_prompt must be provided")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    for attempt in range(max_retries):
        try:
            # Call LLM (JSON requested in prompt, not forced via response_format)
            response_text = call_llm(
                messages=messages,
                response_format=None,  # Don't force JSON mode
                max_tokens=max_tokens,
            )

            # Debug: log raw response for diagnostics
            if not response_text:
                print(f"[llm] WARNING: Empty response from LLM on attempt {attempt + 1}")
            elif len(response_text) < 50:
                print(f"[llm] WARNING: Short response ({len(response_text)} chars): {response_text[:100]!r}")

            # Strip markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            elif response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove closing ```
            response_text = response_text.strip()

            # Parse JSON
            response_data = json.loads(response_text)

            # Validate against schema if provided
            if expected_schema:
                validated = expected_schema(**response_data)
                return validated.model_dump()

            return response_data

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries - 1:
                raise LLMError(f"Failed to get valid JSON response after {max_retries} attempts: {e}")
            # Continue to retry
        except LLMError:
            # Re-raise API errors immediately
            raise

    raise LLMError("Unexpected error in JSON mode call")


# Pipeline stage functions

# =============================================================================
# Story Writing Helpers
# =============================================================================

def _build_character_block(analysis_result: "AnalysisResult") -> str:
    """
    Build a character reference block with pronouns for the story prompt.
    
    Extracts character name, pronouns, role, and traits from analysis data
    and formats them for the LLM to use as a reference.
    
    Example output:
        - Coop: he/him (protagonist - determined, brave)
        - Dash Wheeler: she/her (mentor - confident, supportive)
    """
    lines = []
    for char in analysis_result.characters:
        # Use pronoun from analysis - if it's wrong, fix character extraction
        pronoun = char.pronoun
        
        # Normalize pronoun format (he -> he/him)
        if pronoun and "/" not in pronoun:
            pronoun_map = {
                "she": "she/her", "her": "she/her",
                "he": "he/him", "him": "he/him",
                "they": "they/them", "them": "they/them",
                "it": "it/its",
            }
            pronoun = pronoun_map.get(pronoun.lower(), "they/them")
        
        if not pronoun:
            pronoun = "they/them"  # Safe default
        
        # Build the line
        traits_str = ", ".join(char.traits[:3]) if char.traits else "no traits listed"
        lines.append(f"- {char.name}: {pronoun} ({char.role} - {traits_str})")
    
    if not lines:
        return "No characters identified in analysis."
    
    return "\n".join(lines)


def _extract_catchphrases(
    subtitles_text: str, 
    franchise_db: "Optional[FranchiseData]" = None,
    min_occurrences: int = 1,
) -> list[str]:
    """
    Extract catchphrases from subtitle text using franchise database.
    
    If franchise_db is provided, uses known catchphrases from all characters.
    Also searches for quoted exclamations that appear multiple times.
    
    Args:
        subtitles_text: Raw subtitle text
        franchise_db: Optional franchise database with known catchphrases
        min_occurrences: Minimum times a phrase must appear (default 1 if DB, 2 otherwise)
    
    Returns:
        List of catchphrases found, sorted by occurrence count (most frequent first)
    """
    import re
    from collections import Counter
    
    # Get known catchphrases from franchise DB (generic approach)
    known_catchphrases = []
    if franchise_db:
        known_catchphrases = franchise_db.get_all_catchphrases()
    
    found_catchphrases = []
    text_lower = subtitles_text.lower()
    
    # Check for known catchphrases from DB
    for phrase in known_catchphrases:
        count = text_lower.count(phrase.lower())
        if count >= min_occurrences:
            # Find the original case from the text
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            match = pattern.search(subtitles_text)
            original = match.group(0) if match else phrase
            found_catchphrases.append((original, count))
    
    # Also look for quoted exclamations that appear multiple times
    quoted_pattern = re.compile(r'"([^"]{5,40}[!?])"')
    quotes = quoted_pattern.findall(subtitles_text)
    quote_counts = Counter(q.strip() for q in quotes)
    
    min_quote_occurrences = 2  # Quoted phrases need to appear twice
    for quote, count in quote_counts.items():
        if count >= min_quote_occurrences and quote not in [p[0] for p in found_catchphrases]:
            found_catchphrases.append((quote, count))
    
    # Sort by count (most frequent first) and return just the phrases
    found_catchphrases.sort(key=lambda x: -x[1])
    
    # Return unique phrases
    seen = set()
    result = []
    for phrase, _ in found_catchphrases:
        normalized = phrase.lower().strip('!"\'')
        if normalized not in seen:
            seen.add(normalized)
            result.append(phrase)
    
    # If no results and we have a DB, include top catchphrases from characters
    if not result and franchise_db:
        # Get protagonist/mentor catchphrases as fallback
        for char in franchise_db.characters.values():
            if char.role in ("protagonist", "mentor") and char.catchphrases:
                result.extend(char.catchphrases[:2])
            if len(result) >= 4:
                break
    
    return result[:8]  # Limit to top 8


def _format_catchphrases_block(catchphrases: list[str]) -> str:
    """Format catchphrases into a prompt block."""
    if not catchphrases:
        return "No specific catchphrases identified."
    
    lines = []
    for phrase in catchphrases:
        # Clean up the phrase
        clean = phrase.strip()
        if not clean.endswith(('!', '?', '.')):
            clean += '!'
        lines.append(f'- "{clean}"')
    
    return "\n".join(lines)


# Valid beat types for schema normalization
VALID_BEAT_TYPES = {"setup", "inciting_incident", "rising_action", "climax", "resolution", "other"}
BEAT_TYPE_ALIASES = {
    "crisis": "climax",
    "falling_action": "resolution",
    "exposition": "setup",
    "conflict": "rising_action",
    "denouement": "resolution",
    # Common spacing/hyphen variants
    "inciting incident": "inciting_incident",
    "inciting-incident": "inciting_incident",
    "rising action": "rising_action",
    "rising-action": "rising_action",
    "falling action": "resolution",
    "setup ": "setup",
}


def _normalize_analysis_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess LLM output to fix common schema mismatches.
    
    Handles:
    - String timestamps → floats
    - Unknown beat_type values → valid literals
    - Missing optional fields
    """
    # Normalize beats
    for beat in data.get("beats", []):
        # Fix beat_type
        bt = str(beat.get("beat_type", "other")).lower().strip()
        bt = bt.replace("-", "_").replace(" ", "_")
        beat["beat_type"] = BEAT_TYPE_ALIASES.get(bt, bt if bt in VALID_BEAT_TYPES else "other")
        
        # Fix timestamp_range
        if "timestamp_range" in beat:
            tr = beat["timestamp_range"]
            beat["timestamp_range"] = [_to_float(tr[0]), _to_float(tr[1])] if len(tr) >= 2 else [0.0, 0.0]
    
    # Normalize moments
    for moment in data.get("moments", []):
        # Fix beat_type
        bt = str(moment.get("beat_type", "other")).lower().strip()
        bt = bt.replace("-", "_").replace(" ", "_")
        moment["beat_type"] = BEAT_TYPE_ALIASES.get(bt, bt if bt in VALID_BEAT_TYPES else "other")
        
        # Fix timestamp_range
        if "timestamp_range" in moment:
            tr = moment["timestamp_range"]
            moment["timestamp_range"] = [_to_float(tr[0]), _to_float(tr[1])] if len(tr) >= 2 else [0.0, 0.0]
        
        # Fix screenshot_candidates_s - convert strings to floats
        if "screenshot_candidates_s" in moment:
            moment["screenshot_candidates_s"] = [_to_float(x) for x in moment["screenshot_candidates_s"]]
        
        # Fallback: if all screenshot candidates are 0, generate from timestamp_range
        candidates = moment.get("screenshot_candidates_s", [])
        if all(c == 0.0 for c in candidates) and "timestamp_range" in moment:
            start, end = moment["timestamp_range"]
            if end > start:
                # Generate 3 candidates: start, middle, and 2/3 through
                moment["screenshot_candidates_s"] = [
                    start + (end - start) * 0.1,
                    start + (end - start) * 0.5,
                    start + (end - start) * 0.8,
                ]
    
    return data


def _to_float(value) -> float:
    """Convert value to float, handling strings, time formats, and edge cases."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        cleaned = value.strip()
        
        # Handle time formats like "MM:SS", "HH:MM:SS", "1:30", "01:30:45"
        if ":" in cleaned:
            # Strip any non-numeric suffixes per part (e.g. "01:23s", "00:02:10 (clear)")
            parts = cleaned.split(":")
            try:
                nums: list[float] = []
                for p in parts:
                    m = re.search(r"(\d+(?:\.\d+)?)", p)
                    if not m:
                        nums = []
                        break
                    nums.append(float(m.group(1)))
                if len(nums) == 2:  # MM:SS
                    return nums[0] * 60 + nums[1]
                elif len(nums) == 3:  # HH:MM:SS
                    return nums[0] * 3600 + nums[1] * 60 + nums[2]
            except ValueError:
                pass
        
        # Remove common suffixes and formatting
        for suffix in ["s", "sec", "seconds", " seconds"]:
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
        cleaned = cleaned.replace(",", "").strip()

        # If still not parseable, try extracting the first numeric token.
        m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def analyze_video(video_path: Path, subtitles_text: str) -> AnalysisResult:
    """
    Stage 1: Video Analysis using local MLX-VLM (Qwen3-VL-30B).

    Runs MLX-VLM in a subprocess to avoid Metal initialization conflicts
    with Rich/Typer terminal libraries.

    Args:
        video_path: Path to the video file
        subtitles_text: Raw subtitle text for analysis

    Returns:
        AnalysisResult with beats, characters, and moments
    """
    import subprocess
    import sys
    
    # Get video duration for prompt formatting
    duration_s = ffprobe_duration_s(video_path)
    
    # Format prompts with video metadata
    user_prompt = prompts.VIDEO_ANALYSIS_USER.format(
        video_filename=video_path.name,
        duration_s=duration_s,
        subtitles_text=subtitles_text,
        early_third=duration_s / 3,
        late_third=duration_s * 2 / 3,
    )
    
    # Prepare input for subprocess worker
    input_data = json.dumps({
        "video_path": str(video_path.absolute()),
        "subtitles_text": subtitles_text,
        "system_prompt": prompts.VIDEO_ANALYSIS_SYSTEM,
        "user_prompt": user_prompt,
    })
    
    # Path to MLX worker script
    worker_path = Path(__file__).parent / "mlx_worker.py"
    
    if not worker_path.exists():
        raise LLMError(f"MLX worker script not found: {worker_path}")
    
    try:
        # Run MLX-VLM in isolated subprocess (avoids Metal conflicts)
        result = subprocess.run(
            [sys.executable, str(worker_path)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout for video analysis
        )
        
        # Check for subprocess failure
        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "No error output"
            raise LLMError(f"MLX worker process failed (exit {result.returncode}):\n{stderr}")
        
        # Parse subprocess output
        if not result.stdout.strip():
            raise LLMError("MLX worker returned empty output")
            
        output = json.loads(result.stdout)
        
        if not output.get("success"):
            error_msg = output.get("error", "Unknown error")
            raise LLMError(f"MLX analysis failed: {error_msg}")
        
        # Extract and clean response text
        response_text = output["result"]["text"].strip()
        
        # Strip markdown code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        response_data = json.loads(response_text)
        
        # Normalize data to fix common LLM output issues
        response_data = _normalize_analysis_data(response_data)
        
        # Validate against schema
        return AnalysisResult(**response_data)
        
    except subprocess.TimeoutExpired:
        raise LLMError("MLX video analysis timed out after 15 minutes")
    except json.JSONDecodeError as e:
        raise LLMError(f"Failed to parse MLX response as JSON: {e}")
    except ValidationError as e:
        raise LLMError(f"MLX response doesn't match AnalysisResult schema: {e}")


def _build_franchise_context(franchise_db: "Optional[FranchiseData]") -> str:
    """
    Build franchise-specific context block from database.
    
    Includes character names, vehicles, and other franchise-specific terms
    that should be used in the story instead of generic descriptions.
    """
    if not franchise_db:
        return "No franchise database provided. Extract specific names from subtitles."
    
    lines = [f"FRANCHISE: {franchise_db.franchise_name}"]
    
    # Characters
    char_names = [c.display_name for c in franchise_db.characters.values()]
    if char_names:
        lines.append(f"CHARACTERS: {', '.join(char_names)}")
    
    # Generic Metadata (Vehicles, Locations, Items, etc.)
    for category, items in franchise_db.metadata.items():
        if not items:
            continue
            
        category_label = category.replace("_", " ").upper()
        item_info = []
        for item_id, item_data in items.items():
            # Try to find a good display name
            name = item_data.get("full_name") or item_data.get("name") or item_id.replace("_", " ").title()
            
            # Try to find a descriptive detail (power, type, description, etc.)
            detail = None
            # Check common keys for descriptive text
            for k in ["car_power", "description", "power", "type", "style", "signature_move"]:
                if k in item_data and item_data[k]:
                    detail = item_data[k]
                    break
            
            if detail:
                item_info.append(f"{name} ({detail})")
            else:
                item_info.append(name)
        
        if item_info:
            lines.append(f"{category_label}: {', '.join(item_info[:6])}")
    
    return "\n".join(lines)


def write_story(
    analysis_result: AnalysisResult, 
    subtitles_text: str,
    franchise_db: "FranchiseData",
    target_pages: int = 22,
) -> str:
    """
    Stage 2: Story Writing.

    Generates a children's picture book story from video analysis data.
    
    Key improvements in this version:
    - Builds character reference block with ENFORCED pronouns
    - Extracts and includes signature catchphrases from franchise DB + subtitles
    - Passes structured franchise data for specific names
    - Injects franchise-specific style examples from franchise JSON
    - Generates content scaled to target page count

    Args:
        analysis_result: Analysis from stage 1 (with characters, beats, moments)
        subtitles_text: Raw subtitle text for dialogue/catchphrase extraction
        franchise_db: REQUIRED franchise database for style examples
        target_pages: Target number of pages (affects paragraph count)

    Returns:
        Complete story text in Markdown format with proper pronouns,
        catchphrases, and narrative structure.
    
    Raises:
        ValueError: If franchise_db is not provided
    """
    if not franchise_db:
        raise ValueError("franchise_db is required for write_story")
    
    # Build character reference block with pronouns
    character_block = _build_character_block(analysis_result)
    
    # Extract catchphrases from subtitles using franchise DB
    catchphrases = _extract_catchphrases(subtitles_text, franchise_db)
    catchphrases_block = _format_catchphrases_block(catchphrases)
    
    # Build franchise-specific context
    franchise_context = _build_franchise_context(franchise_db)
    
    # Get franchise-specific style examples and rules for prompt injection
    franchise_style_examples = franchise_db.get_story_style_examples()
    franchise_style_rules = franchise_db.get_style_rules_prompt()
    franchise_pronoun_example = franchise_db.get_pronoun_example()
    
    # Format system prompt with franchise examples
    system_prompt = prompts.STORY_WRITING_SYSTEM.format(
        franchise_style_examples=franchise_style_examples,
        franchise_style_rules=franchise_style_rules,
        franchise_pronoun_example=franchise_pronoun_example,
    )
    
    # Calculate paragraph targets based on page count
    # Each page typically has 1-2 paragraphs, so target slightly more paragraphs than pages
    target_paragraphs_min = max(target_pages - 4, 16)
    target_paragraphs_max = target_pages + 4
    
    # Format the user prompt with all data blocks
    user_prompt = prompts.STORY_WRITING_USER.format(
        character_block=character_block,
        catchphrases_block=catchphrases_block,
        franchise_context=franchise_context,
        analysis_json=json.dumps(analysis_result.model_dump(), indent=2),
        subtitles_text=subtitles_text,
        target_paragraphs_min=target_paragraphs_min,
        target_paragraphs_max=target_paragraphs_max,
    )

    return call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.llm.temperature_story,
    )


def paginate_story(
    story_text: str, 
    analysis_result: AnalysisResult,
    franchise_db: "FranchiseData",
    target_pages: int = 22,
    min_pages: int = 18,
    max_pages: int = 40,
    words_per_page: int = 45,
) -> BookSpec:
    """
    Stage 3: Pagination - convert story into paginated book structure.

    Args:
        story_text: Generated story from stage 2
        analysis_result: Analysis from stage 1
        franchise_db: REQUIRED franchise database for pagination examples
        target_pages: Target number of pages
        min_pages: Minimum allowed pages
        max_pages: Maximum allowed pages
        words_per_page: Target words per page

    Returns:
        BookSpec with pages array
    
    Raises:
        ValueError: If franchise_db is not provided
    """
    if not franchise_db:
        raise ValueError("franchise_db is required for paginate_story")
    
    # Get franchise-specific pagination examples
    franchise_pagination_examples = franchise_db.get_pagination_examples()
    
    # Format system prompt with pagination parameters
    system_prompt = prompts.PAGINATION_SYSTEM.format(
        target_pages=target_pages,
        min_pages=min_pages,
        max_pages=max_pages,
        words_per_page=words_per_page,
    )
    
    user_prompt = prompts.PAGINATION_USER.format(
        story_text=story_text,
        analysis_json=json.dumps(analysis_result.model_dump(), indent=2),
        target_pages=target_pages,
        min_pages=min_pages,
        max_pages=max_pages,
        words_per_page=words_per_page,
        franchise_pagination_examples=franchise_pagination_examples,
    )

    # Use higher max_tokens for pagination - more pages needs more tokens
    max_tokens = max(16384, target_pages * 800)  # Scale with page count
    
    response_data = call_with_json_mode(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_schema=BookSpec,
        max_tokens=max_tokens,
    )

    return BookSpec(**response_data)


def enforce_alternating_layout(book_spec: BookSpec) -> BookSpec:
    """
    Post-pagination step: Enforce deterministic alternating layout pattern.
    
    Picture books benefit from a consistent visual rhythm where the image
    position alternates between pages. This removes layout decisions from
    the LLM (which wastes tokens on a deterministic rule) and ensures
    professional consistency.
    
    Rule:
        - Odd pages (1, 3, 5, ...): image_top
        - Even pages (2, 4, 6, ...): image_bottom
    
    This creates a natural "zig-zag" reading flow that keeps the book
    visually engaging without being chaotic.
    
    Args:
        book_spec: The paginated book specification
        
    Returns:
        BookSpec with enforced layout_hint values
    """
    changed_count = 0
    
    for page in book_spec.pages:
        new_layout = "image_top" if page.page_index % 2 == 1 else "image_bottom"
        if page.layout_hint != new_layout:
            changed_count += 1
        page.layout_hint = new_layout
    
    if changed_count > 0:
        print(f"[layout] Enforced alternating layout for {changed_count}/{len(book_spec.pages)} pages")
    
    return book_spec


def enforce_chronological_moments(book_spec: BookSpec, analysis_result: AnalysisResult) -> BookSpec:
    """
    Post-pagination step: Ensure moment_ids are in chronological order.
    
    The LLM sometimes assigns moment_ids that go backwards in time.
    This function fixes violations by advancing to the next available moment.
    
    IMPORTANT: We track prev_timestamp as the moment END (not start) because
    frame selection can pick any frame within [moment_start, moment_end].
    If we only tracked moment_start, overlapping moments could pass this check
    but fail during frame selection when a late frame is chosen.
    
    Args:
        book_spec: The paginated book specification
        analysis_result: Video analysis with moments
        
    Returns:
        BookSpec with chronologically ordered moment_ids
        
    Raises:
        ValueError: If no valid moment can be found for a page (moment exhaustion)
    """
    # Build moment lookup with timestamps
    moments_by_id = {m.moment_id: m for m in analysis_result.moments}
    sorted_moments = sorted(analysis_result.moments, key=lambda m: m.timestamp_range[0])
    
    fixed_count = 0
    prev_moment_end = -1.0  # Track END of previous moment, not start
    
    for page in book_spec.pages:
        if not page.moment_id:
            continue
            
        moment = moments_by_id.get(page.moment_id)
        if not moment:
            continue
        
        moment_start = moment.timestamp_range[0]
        moment_end = moment.timestamp_range[1]
        
        # Check for chronological violation: moment must START after previous moment ENDED
        # This prevents overlapping moments which cause frame selection failures
        if moment_start < prev_moment_end:
            # Find next moment that starts after prev_moment_end
            found_replacement = False
            for m in sorted_moments:
                if m.timestamp_range[0] >= prev_moment_end:
                    old_id = page.moment_id
                    page.moment_id = m.moment_id
                    moment_end = m.timestamp_range[1]
                    fixed_count += 1
                    print(
                        f"[chronology] Page {page.page_index}: {old_id} → {m.moment_id} "
                        f"(was {moments_by_id[old_id].timestamp_range[0]:.1f}s, needed >= {prev_moment_end:.1f}s)"
                    )
                    found_replacement = True
                    break
            
            # Fail fast if no valid moment exists - don't let this silently cause
            # frame selection to fail hours later
            if not found_replacement:
                raise ValueError(
                    f"Page {page.page_index}: Cannot find moment starting after {prev_moment_end:.1f}s. "
                    f"Video has {len(sorted_moments)} moments but pagination requires non-overlapping "
                    f"moments for {len(book_spec.pages)} pages. Consider reducing page count."
                )
        
        prev_moment_end = moment_end  # Track END for next iteration
    
    if fixed_count > 0:
        print(f"[chronology] Fixed {fixed_count} chronological violations")
    else:
        print(f"[chronology] All {len(book_spec.pages)} pages in chronological order ✓")
    
    return book_spec


def generate_visual_targets(
    pages: list[PageSpec], 
    analysis_result: AnalysisResult,
    franchise_db: "FranchiseData",
) -> list[VisualTarget]:
    """
    Stage 4a (Milestone C): Generate per-page visual targets + anchors (MiMo).

    We do this as a separate explicit step so frame selection can be driven by a
    crisp "what must be visible" spec rather than relying on timestamps alone.
    
    Args:
        pages: List of page specs to generate targets for
        analysis_result: Video analysis result
        franchise_db: REQUIRED franchise data for visual style guidance
    
    Raises:
        ValueError: If franchise_db is not provided
    """
    if not franchise_db:
        raise ValueError("franchise_db is required for generate_visual_targets")
    
    pages_json = json.dumps([p.model_dump() for p in pages], indent=2, ensure_ascii=False)
    analysis_json = json.dumps(analysis_result.model_dump(), indent=2, ensure_ascii=False)

    # Get franchise-specific visual guidance and examples
    franchise_visual_guidance = franchise_db.get_visual_guidance_prompt()
    franchise_visual_target_examples = franchise_db.get_visual_target_examples()
    
    user_prompt = prompts.VISUAL_TARGETS_USER.format(
        pages_json=pages_json,
        analysis_json=analysis_json,
        franchise_visual_guidance=franchise_visual_guidance,
        franchise_visual_target_examples=franchise_visual_target_examples,
    )

    # Build system prompt with franchise visual guidance
    system_prompt = prompts.VISUAL_TARGETS_SYSTEM
    if franchise_visual_guidance:
        system_prompt = system_prompt + "\n\n" + franchise_visual_guidance

    response_data = call_with_json_mode(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_schema=VisualTargetsResult,
    )

    parsed = VisualTargetsResult(**response_data)

    # Opinionated validation: must provide 1 target per page index.
    expected = {p.page_index for p in pages}
    got = {t.page_index for t in parsed.targets}
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise LLMError(f"Visual target generation mismatch. missing={missing} extra={extra}")

    # Return in page order
    by_index = {t.page_index: t for t in parsed.targets}
    return [by_index[p.page_index] for p in pages]
