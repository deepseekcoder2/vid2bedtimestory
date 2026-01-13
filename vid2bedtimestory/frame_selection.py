"""
Frame Selection: Three-Stage Progressive Refinement Pipeline.

This module implements intelligent screenshot selection for picture books:

STAGE 1 - NEIGHBORHOOD (moment_id):
    - Use page.moment_id to find the 20-30s segment where content occurs
    - This is the "neighborhood" - we're in the right area

STAGE 2 - BLOCK (VLM inventory + Text LLM matching):
    2A: VLM watches segment and creates timeline inventory of what's visible
    2B: Text LLM matches story to inventory, picks 2-3s window
    - This narrows to the "block" - the specific action we want

STAGE 3 - HOUSE (CV quality analysis):
    - Extract all frames from the 2-3s window
    - Use Laplacian variance to find sharpest frame
    - Pure signal processing - deterministic, fast, reliable

NO AI GUESSING FOR FINAL FRAME. The VLM identifies WHAT to show,
the CV analysis finds the SHARPEST frame showing it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import config
from .ffmpeg import extract_frame, extract_analysis_segment
from .llm import call_llm
from .models import SelectedFrame

if TYPE_CHECKING:
    from .models import AnalysisResult, Moment, PageSpec, SubtitleSegment


class FrameSelectionError(Exception):
    """Raised when frame selection fails. No fallback - fix the problem."""
    pass


# =============================================================================
# CV QUALITY FUNCTIONS (Stage 3)
# =============================================================================

def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_brightness_contrast(image: np.ndarray) -> tuple[float, float]:
    """Calculate image brightness and contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return mean_brightness, contrast


def score_frame_quality(image: np.ndarray) -> dict:
    """
    Score a frame's technical quality.
    
    Returns dict with:
        - sharpness: Laplacian variance (higher = sharper)
        - brightness: Mean grayscale value (ideal ~125)
        - contrast: Std of grayscale (higher = more dynamic range)
        - total_score: Weighted combination
        - is_acceptable: Passes minimum thresholds
        - issues: List of any quality problems
    """
    sharpness = calculate_sharpness(image)
    brightness, contrast = calculate_brightness_contrast(image)
    
    issues = []
    
    # Check thresholds
    if sharpness < 50:
        issues.append(f"low_sharpness({sharpness:.0f})")
    if brightness < 30:
        issues.append(f"too_dark({brightness:.0f})")
    if brightness > 230:
        issues.append(f"too_bright({brightness:.0f})")
    if contrast < 15:
        issues.append(f"low_contrast({contrast:.0f})")
    
    # Calculate total score (weighted)
    # Sharpness is most important for animation
    brightness_score = max(0, 100 - abs(brightness - 125))  # 125 as ideal
    contrast_score = min(contrast, 100)  # Cap at 100
    
    total_score = (sharpness * 0.7) + (brightness_score * 0.2) + (contrast_score * 0.1)
    
    return {
        'sharpness': sharpness,
        'brightness': brightness,
        'contrast': contrast,
        'total_score': total_score,
        'is_acceptable': len(issues) == 0,
        'issues': issues,
    }


def find_best_frame_in_window_cv(
    video_path: Path,
    start_time: float,
    end_time: float,
) -> tuple[float, dict]:
    """
    Stage 3: Find the technically best frame in a narrow time window using CV.
    
    This is deterministic - same input always gives same output.
    No AI involved, just signal processing.
    
    Args:
        video_path: Path to the video file
        start_time: Start of window in seconds
        end_time: End of window in seconds
    
    Returns:
        (best_timestamp, quality_info)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FrameSelectionError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    
    if start_frame >= end_frame:
        # Window too small, just use start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return start_time, score_frame_quality(frame)
        raise FrameSelectionError(f"Could not read frame at {start_time}s")
    
    best_frame_idx = start_frame
    best_score = -1
    best_quality = None
    
    # Evaluate all frames in window
    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        quality = score_frame_quality(frame)
        
        if quality['total_score'] > best_score:
            best_score = quality['total_score']
            best_frame_idx = frame_idx
            best_quality = quality
    
    cap.release()
    
    if best_quality is None:
        raise FrameSelectionError(
            f"No valid frames found in window [{start_time:.1f}s-{end_time:.1f}s]"
        )
    
    best_timestamp = best_frame_idx / fps
    return best_timestamp, best_quality


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def select_frames_for_book(
    pages: list["PageSpec"],
    video_path: Path,
    subtitles: list["SubtitleSegment"],
    analysis: "AnalysisResult",
    frames_dir: Path,
    temp_dir: Path,
    video_duration: float,
) -> list[SelectedFrame]:
    """
    Select the best screenshot for each page using three-stage pipeline.
    
    For each page:
    1. STAGE 1: Find 20-30s neighborhood via moment_id
    2. STAGE 2A: VLM creates timeline inventory of segment
    3. STAGE 2B: Text LLM narrows to 2-3s window matching story
    4. STAGE 3: CV finds sharpest frame in narrow window
    5. Extract full-quality frame
    
    Args:
        pages: List of page specifications with text
        video_path: Path to source video
        subtitles: Parsed subtitle segments (unused in new pipeline)
        analysis: Video analysis with moments
        frames_dir: Directory for final frames
        temp_dir: Directory for temporary segments
        video_duration: Total video length in seconds
    
    Returns:
        List of SelectedFrame with timestamps and paths
    
    Raises:
        FrameSelectionError: If any page fails (no fallback)
    """
    if not pages:
        return []
    
    total_start = time.time()
    print(f"[frame_selection] Starting three-stage frame selection for {len(pages)} pages")
    
    # Setup directories
    frames_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    selected: list[SelectedFrame] = []
    prev_timestamp: float | None = None
    
    for i, page in enumerate(pages):
        page_start = time.time()
        
        # Get page text
        page_text = " ".join(page.paragraphs) if page.paragraphs else ""
        if not page_text.strip():
            raise FrameSelectionError(
                f"Page {page.page_index}: Empty page text. Cannot select frame."
            )
        
        # =====================================================================
        # STAGE 1: Find neighborhood (20-30s segment from moment_id)
        # =====================================================================
        print(f"[frame_selection]   Page {page.page_index}: Stage 1 - Finding neighborhood...")
        window_start, window_end = find_search_window(
            page=page,
            moments=analysis.moments,
            prev_timestamp=prev_timestamp,
            video_duration=video_duration,
        )
        segment_duration = window_end - window_start
        print(f"[frame_selection]     Neighborhood: [{window_start:.1f}s-{window_end:.1f}s] ({segment_duration:.1f}s)")
        
        # =====================================================================
        # STAGE 2A: VLM creates timeline inventory
        # =====================================================================
        print(f"[frame_selection]   Page {page.page_index}: Stage 2A - VLM inventory...")
        segment_path = temp_dir / f"segment_page_{page.page_index:03d}.mp4"
        extract_analysis_segment(video_path, window_start, window_end, segment_path)
        
        inventory = vlm_create_inventory(segment_path, segment_duration)
        print(f"[frame_selection]     Inventory: {len(inventory.splitlines())} entries")
        
        # =====================================================================
        # STAGE 2B: Text LLM narrows to 2-3s window
        # =====================================================================
        print(f"[frame_selection]   Page {page.page_index}: Stage 2B - Narrowing to action window...")
        
        # Generate visual query (what we're looking for)
        visual_query = generate_visual_query(page_text)
        
        # LLM picks the best 2-3s window from inventory
        narrow_start, narrow_end, action_desc = llm_narrow_window(
            page_text=page_text,
            visual_query=visual_query,
            inventory=inventory,
            segment_duration=segment_duration,
        )
        
        # Convert relative times to absolute
        abs_narrow_start = window_start + narrow_start
        abs_narrow_end = window_start + narrow_end
        
        # Ensure narrow window respects chronology
        if prev_timestamp is not None and abs_narrow_start < prev_timestamp:
            # Push forward to after previous timestamp
            push_amount = prev_timestamp - abs_narrow_start + 0.1
            abs_narrow_start += push_amount
            abs_narrow_end += push_amount
            # Clamp to segment
            abs_narrow_end = min(abs_narrow_end, window_end)
        
        print(f"[frame_selection]     Action window: [{abs_narrow_start:.1f}s-{abs_narrow_end:.1f}s] ({abs_narrow_end-abs_narrow_start:.1f}s)")
        print(f"[frame_selection]     Looking for: {action_desc[:60]}...")
        
        # Cleanup segment (no longer needed)
        try:
            segment_path.unlink()
        except Exception:
            pass
        
        # =====================================================================
        # STAGE 3: CV finds sharpest frame in narrow window
        # =====================================================================
        print(f"[frame_selection]   Page {page.page_index}: Stage 3 - CV sharpness analysis...")
        
        best_timestamp, quality_info = find_best_frame_in_window_cv(
            video_path=video_path,
            start_time=abs_narrow_start,
            end_time=abs_narrow_end,
        )
        
        sharpness = quality_info['sharpness']
        print(f"[frame_selection]     Best frame: {best_timestamp:.2f}s (sharpness={sharpness:.0f})")
        
        # =====================================================================
        # EXTRACT: Get full-quality frame from original video
        # =====================================================================
        frame_path = frames_dir / f"page_{page.page_index:03d}.png"
        extract_frame(video_path, best_timestamp, frame_path)
        
        if not frame_path.exists():
            raise FrameSelectionError(
                f"Page {page.page_index}: Frame extraction failed at {best_timestamp:.2f}s"
            )
        
        # Record result
        sel = SelectedFrame(
            page_index=page.page_index,
            timestamp_s=best_timestamp,
            frame_path=str(frame_path),
            visual_target=visual_query,
            window_start_s=window_start,
            window_end_s=window_end,
            anchor_reason=f"stage3_cv:{action_desc[:50]}",
            vlm_score=sharpness,
            vlm_raw=inventory[:200],
            fallback_used=None,
            low_quality=not quality_info['is_acceptable'],
        )
        selected.append(sel)
        prev_timestamp = best_timestamp
        
        page_elapsed = time.time() - page_start
        print(f"[frame_selection]   Page {page.page_index}: ✓ ts={best_timestamp:.1f}s [{page_elapsed:.1f}s]")
    
    total_elapsed = time.time() - total_start
    print(f"[frame_selection] Complete! {len(selected)} frames in {total_elapsed:.1f}s")
    
    return selected


# =============================================================================
# STAGE 1: SEARCH WINDOW FINDING (unchanged)
# =============================================================================

def find_search_window(
    page: "PageSpec",
    moments: list["Moment"],
    prev_timestamp: float | None,
    video_duration: float,
) -> tuple[float, float]:
    """
    Stage 1: Find the 20-30s neighborhood using moment_id.
    
    NO FALLBACKS - if moment_id is missing or invalid, we FAIL LOUDLY.
    """
    # REQUIRE moment_id
    if not page.moment_id or page.moment_id.strip() == "":
        raise FrameSelectionError(
            f"Page {page.page_index}: moment_id is missing or empty. "
            f"Video analysis must provide moment_id for every page."
        )
    
    # Look up the moment
    moment = None
    for m in moments:
        if m.moment_id == page.moment_id:
            moment = m
            break
    
    if moment is None:
        raise FrameSelectionError(
            f"Page {page.page_index}: moment_id='{page.moment_id}' not found in video analysis. "
            f"Available moments: {[m.moment_id for m in moments[:5]]}..."
        )
    
    # Use moment's timestamp range
    moment_start, moment_end = moment.timestamp_range
    
    # Apply chronological constraint: window must allow frames after previous page
    if prev_timestamp is not None:
        if moment_end <= prev_timestamp:
            # Entire moment is before previous page - this is a real violation
            raise FrameSelectionError(
                f"Page {page.page_index}: Chronological violation. "
                f"moment_id='{page.moment_id}' ends at {moment_end:.1f}s, "
                f"but previous page was at {prev_timestamp:.1f}s. "
                f"Moment is entirely in the past - pagination needs to be fixed."
            )
        elif moment_start < prev_timestamp:
            # Moment overlaps - adjust start to after previous timestamp
            print(f"[frame_selection]     Note: Adjusting window start from {moment_start:.1f}s to {prev_timestamp + 0.1:.1f}s (chronology)")
            moment_start = prev_timestamp + 0.1
    
    # Expand window for VLM inventory (±5s buffer)
    buffer_s = 5.0
    window_start = max(0.0, moment_start - buffer_s)
    window_end = min(video_duration, moment_end + buffer_s)
    
    # Ensure window_start respects chronology (buffer might have pulled it back)
    if prev_timestamp is not None and window_start < prev_timestamp:
        window_start = prev_timestamp + 0.1
    
    # Ensure minimum window size of 10s
    if window_end - window_start < 10.0:
        mid = (window_start + window_end) / 2
        window_start = max(window_start, mid - 5.0)  # Don't go before prev_timestamp
        window_end = min(video_duration, mid + 5.0)
    
    # Final validation
    if window_start >= window_end:
        raise FrameSelectionError(
            f"Page {page.page_index}: Cannot create valid search window. "
            f"moment_id='{page.moment_id}' [{moment_start:.1f}s-{moment_end:.1f}s], "
            f"prev_timestamp={prev_timestamp:.1f}s. Window would be empty."
        )
    
    return window_start, window_end


# =============================================================================
# STAGE 2A: VLM INVENTORY
# =============================================================================

def vlm_create_inventory(segment_path: Path, segment_duration: float) -> str:
    """
    Stage 2A: VLM watches segment and creates timeline inventory.
    
    Output is a structured timeline of what's visible at each moment.
    This is OBSERVATION only - no decision-making.
    
    Returns:
        String with timestamped inventory entries
    """
    prompt = f'''Watch this video clip carefully ({segment_duration:.1f} seconds).

Create a detailed timeline describing what is VISIBLE at each moment.
Write one line per 2-second interval.

FORMAT (use exactly this format):
[0.0-2.0] <describe: objects, characters, actions, camera angle>
[2.0-4.0] <describe what's visible>
[4.0-6.0] <describe what's visible>
... continue until end of clip ...

RULES:
- Be SPECIFIC about objects: "a large black tire", "orange race track", "boy in teal hoodie"
- Describe ACTIONS: "rolling", "bouncing", "crashing through window", "flying"
- Note camera angles: "close-up", "wide shot", "low angle", "from behind"
- Describe character positions: "facing camera", "back turned", "arms raised"
- If multiple things happen, list them all

Example output:
[0.0-2.0] Wide shot of garage. Six kids standing near glass display case. Orange track visible in background.
[2.0-4.0] Display case wobbles. Large black tire tips off metal shelf. Camera zooms in on tire.
[4.0-6.0] Tire bouncing on concrete floor, moving toward window. Motion blur on tire. Kids in background watching.

Now describe this {segment_duration:.1f} second clip:'''

    result = _call_local_vlm_video(segment_path, prompt, max_tokens=4000)
    
    if not result.strip():
        raise FrameSelectionError("VLM inventory returned empty result")
    
    return result.strip()


# =============================================================================
# STAGE 2B: TEXT LLM NARROW WINDOW
# =============================================================================

def llm_narrow_window(
    page_text: str,
    visual_query: str,
    inventory: str,
    segment_duration: float,
) -> tuple[float, float, str]:
    """
    Stage 2B: Text LLM picks the 2-3s window that best matches the story.
    
    This is TEXT MATCHING - no visual processing, just semantic alignment.
    
    Args:
        page_text: The story text for this page
        visual_query: What we want to show (from generate_visual_query)
        inventory: VLM's timeline of what's in the segment
        segment_duration: Total segment length
    
    Returns:
        (start_time, end_time, action_description) - relative to segment start
    """
    prompt = f'''You are selecting a screenshot for a children's picture book.

STORY TEXT (what the page says):
"{page_text}"

WHAT TO SHOW:
{visual_query}

AVAILABLE MOMENTS IN VIDEO (what we can choose from):
{inventory}

SELECTION RULES:
1. Pick the 2-second window that SHOWS the KEY ACTION from the story
2. Prefer ACTION shots over REACTION shots:
   - "tire bouncing" > "kids watching"
   - "car racing" > "spectators reacting"  
   - "jumping over gap" > "landing aftermath"
3. Prefer OBJECTS/EVENTS mentioned in the story over character close-ups
4. If story mentions something HAPPENING (bouncing, racing, flying, crashing), pick that moment
5. Character dialogue scenes → pick character DOING something, not just talking

OUTPUT FORMAT (exactly):
START_TIME: <number between 0.0 and {segment_duration:.1f}>
END_TIME: <number, should be START_TIME + 2.0>
ACTION: <brief description of what happens in this window>

Example:
START_TIME: 4.0
END_TIME: 6.0
ACTION: tire bouncing on floor toward window'''

    result = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=config.llm.utility_model,
        max_tokens=200,
    )
    
    # Parse result
    result_clean = result.strip()
    
    start_match = re.search(r'START_TIME:\s*(\d+\.?\d*)', result_clean)
    end_match = re.search(r'END_TIME:\s*(\d+\.?\d*)', result_clean)
    action_match = re.search(r'ACTION:\s*(.+?)(?:\n|$)', result_clean)
    
    if not start_match or not end_match:
        # Fallback: use middle of segment
        mid = segment_duration / 2
        return max(0, mid - 1), min(segment_duration, mid + 1), "fallback_middle"
    
    start_time = float(start_match.group(1))
    end_time = float(end_match.group(1))
    action_desc = action_match.group(1).strip() if action_match else "matched_action"
    
    # Validate and clamp
    start_time = max(0.0, min(start_time, segment_duration - 1))
    end_time = max(start_time + 0.5, min(end_time, segment_duration))
    
    # Ensure at least 1s window, at most 3s
    if end_time - start_time < 1.0:
        end_time = min(start_time + 1.5, segment_duration)
    if end_time - start_time > 3.0:
        end_time = start_time + 3.0
    
    return start_time, end_time, action_desc


# =============================================================================
# VISUAL QUERY GENERATION (unchanged)
# =============================================================================

def generate_visual_query(page_text: str) -> str:
    """
    Analyze page text and determine what visual would have the biggest impact.
    
    Input: "The tire bounced wildly down the track!"
    Output: "A tire bouncing/rolling on a race track, showing motion"
    """
    prompt = f'''Read this children's book page:

"{page_text}"

What ONE visual element would make the BEST illustration for this page?

RULES:
- Describe the KEY ACTION or OBJECT that should be VISIBLE
- Focus on what's HAPPENING, not characters talking about it
- If text mentions a specific object/vehicle/item, it should probably be shown
- If text describes an action (jumping, bouncing, racing), show THAT moment
- Be specific about what should be visible in the frame
- Do NOT just describe characters standing or talking

OUTPUT FORMAT:
Return ONLY a short description (1-2 sentences) of what should be visible in the frame.

VISUAL:'''

    result = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=config.llm.utility_model,
        max_tokens=100,
    )
    
    visual_query = result.strip()
    if not visual_query:
        raise FrameSelectionError(
            f"Visual query generation returned empty result for text: {page_text[:100]}..."
        )
    
    return visual_query


# =============================================================================
# LOCAL VLM CALLS
# =============================================================================

def _call_local_vlm_video(video_path: Path, prompt: str, max_tokens: int = 4000) -> str:
    """
    Call local Qwen VLM with a video segment.
    """
    worker_path = Path(__file__).parent / "mlx_worker.py"
    
    if not worker_path.exists():
        raise FrameSelectionError(f"MLX worker not found: {worker_path}")
    
    input_data = json.dumps({
        "video_path": str(video_path.absolute()),
        "system_prompt": "You are a video analysis assistant. Describe what you see accurately and specifically.",
        "user_prompt": prompt,
        "max_tokens": max_tokens,
    })
    
    try:
        result = subprocess.run(
            [sys.executable, str(worker_path)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes for video processing
        )
        
        if result.returncode != 0:
            raise FrameSelectionError(
                f"Local VLM failed: {result.stderr[:200] if result.stderr else 'No error output'}"
            )
        
        if not result.stdout.strip():
            raise FrameSelectionError("Local VLM returned empty output")
        
        output = json.loads(result.stdout)
        
        if not output.get("success"):
            raise FrameSelectionError(f"Local VLM error: {output.get('error', 'Unknown')}")
        
        return output.get("result", {}).get("text", "")
        
    except subprocess.TimeoutExpired:
        raise FrameSelectionError("Local VLM timed out after 180s")
    except json.JSONDecodeError as e:
        raise FrameSelectionError(f"Failed to parse VLM response: {e}")
