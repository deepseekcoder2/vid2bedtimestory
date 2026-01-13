"""
Pipeline stage management with Makefile-style dependency tracking.

Each stage has defined inputs and outputs. A stage runs if:
1. Any output is missing, OR
2. Any input is newer than any output

This is zero-overhead (just stat calls) and self-healing.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


# =============================================================================
# STALE ARTIFACT PATTERNS
# =============================================================================
# These files/patterns are from old versions and should be cleaned up.
# Add patterns here when renaming artifacts or deprecating features.

STALE_ARTIFACTS = [
    # Old analysis naming
    "analysis_v2.json",
    # Old pagination iterations  
    "pages_v2.json",
    "pages_v3.json", 
    "pages_final.json",
    # Old VLM build logs
    "vlm_build.log",
]

STALE_DIRECTORIES = [
    # Candidate frames (should be cleaned after run)
    "frames/_candidates",
]


class Stage(str, Enum):
    """Pipeline stages in execution order."""
    SUBTITLES = "subtitles"
    ANALYSIS = "analysis"
    STORY = "story"
    PAGINATION = "pagination"
    SCREENSHOTS = "screenshots"
    PDF = "pdf"
    COMPRESS = "compress"


@dataclass
class StageSpec:
    """Specification for a pipeline stage."""
    name: Stage
    inputs: Callable[[Path, Path], list[Path]]  # (video_path, artifacts_dir) -> input paths
    outputs: Callable[[Path, Path], list[Path]]  # (video_path, artifacts_dir) -> output paths
    description: str


def _get_stage_specs() -> dict[Stage, StageSpec]:
    """
    Define the dependency graph for all stages.
    
    Each stage specifies its inputs and outputs as functions of
    (video_path, artifacts_dir) to handle dynamic paths.
    """
    return {
        Stage.SUBTITLES: StageSpec(
            name=Stage.SUBTITLES,
            inputs=lambda v, a: [v],
            outputs=lambda v, a: [a / "subtitles.srt", a / "subtitles.json"],
            description="Extract subtitles from video",
        ),
        Stage.ANALYSIS: StageSpec(
            name=Stage.ANALYSIS,
            inputs=lambda v, a: [v, a / "subtitles.json"],
            outputs=lambda v, a: [a / "analysis.json"],
            description="Analyze video with VLM pipeline",
        ),
        Stage.STORY: StageSpec(
            name=Stage.STORY,
            inputs=lambda v, a: [a / "analysis.json", a / "subtitles.json"],
            outputs=lambda v, a: [a / "story.md"],
            description="Generate story from analysis",
        ),
        Stage.PAGINATION: StageSpec(
            name=Stage.PAGINATION,
            inputs=lambda v, a: [a / "story.md", a / "analysis.json"],
            outputs=lambda v, a: [a / "pages.json"],
            description="Paginate story into book layout",
        ),
        Stage.SCREENSHOTS: StageSpec(
            name=Stage.SCREENSHOTS,
            inputs=lambda v, a: [v, a / "pages.json", a / "analysis.json"],
            outputs=lambda v, a: [a / "selected_frames.json", a / "frames"],
            description="Select and extract screenshot frames",
        ),
        Stage.PDF: StageSpec(
            name=Stage.PDF,
            inputs=lambda v, a: [a / "pages.json", a / "frames"],
            outputs=lambda v, a: [],  # Output path is user-specified, not in artifacts
            description="Render final PDF",
        ),
        Stage.COMPRESS: StageSpec(
            name=Stage.COMPRESS,
            inputs=lambda v, a: [],
            outputs=lambda v, a: [],
            description="Compress PDF for digital transfer",
        ),
    }


STAGE_SPECS = _get_stage_specs()
STAGE_ORDER = list(Stage)


def should_run_stage(
    stage: Stage,
    video_path: Path,
    artifacts_dir: Path,
    force_from: Optional[Stage] = None,
) -> tuple[bool, str]:
    """
    Determine if a stage needs to run based on file modification times.
    
    Args:
        stage: The stage to check
        video_path: Path to source video
        artifacts_dir: Path to artifacts directory
        force_from: If set, force rebuild from this stage onwards AND skip earlier stages
        
    Returns:
        (should_run, reason) tuple
    """
    spec = STAGE_SPECS[stage]
    inputs = spec.inputs(video_path, artifacts_dir)
    outputs = spec.outputs(video_path, artifacts_dir)
    
    # Handle --rebuild-from flag
    if force_from is not None:
        force_idx = STAGE_ORDER.index(force_from)
        stage_idx = STAGE_ORDER.index(stage)
        
        # Stages BEFORE force_from: skip (use existing artifacts)
        if stage_idx < force_idx:
            # But verify outputs exist
            missing = [o for o in outputs if not o.exists()]
            if missing:
                return True, f"required by --rebuild-from but missing: {[o.name for o in missing]}"
            return False, f"skipped (before {force_from.value})"
        
        # Stages AT or AFTER force_from: force run
        return True, f"forced (--rebuild-from={force_from.value})"
    
    # No outputs defined = always run (e.g., PDF stage)
    if not outputs:
        return True, "no tracked outputs"
    
    # Check if any output is missing
    missing_outputs = [o for o in outputs if not o.exists()]
    if missing_outputs:
        missing_names = [o.name for o in missing_outputs]
        return True, f"missing: {', '.join(missing_names)}"
    
    # Check if any input is missing (error case)
    existing_inputs = [i for i in inputs if i.exists()]
    if not existing_inputs:
        return True, "no inputs found"
    
    # Get modification times
    newest_input_mtime = max(
        _get_mtime(p) for p in inputs if p.exists()
    )
    oldest_output_mtime = min(
        _get_mtime(p) for p in outputs if p.exists()
    )
    
    # If any input is newer than oldest output, rebuild
    if newest_input_mtime > oldest_output_mtime:
        return True, "inputs newer than outputs"
    
    return False, "up to date"


def _get_mtime(path: Path) -> float:
    """Get modification time, handling directories by checking contents."""
    if path.is_dir():
        # For directories, use the newest file inside
        files = list(path.rglob("*"))
        if files:
            return max(f.stat().st_mtime for f in files if f.is_file())
        return 0.0
    return path.stat().st_mtime


def get_pipeline_status(
    video_path: Path,
    artifacts_dir: Path,
    force_from: Optional[Stage] = None,
) -> dict[Stage, tuple[bool, str]]:
    """
    Get run/skip status for all stages.
    
    Returns:
        Dict mapping stage -> (should_run, reason)
    """
    status = {}
    for stage in STAGE_ORDER:
        status[stage] = should_run_stage(stage, video_path, artifacts_dir, force_from)
    return status


def print_pipeline_plan(
    video_path: Path,
    artifacts_dir: Path,
    force_from: Optional[Stage] = None,
) -> None:
    """Print what the pipeline will do."""
    status = get_pipeline_status(video_path, artifacts_dir, force_from)
    
    print("\nPipeline execution plan:")
    print("-" * 60)
    for stage in STAGE_ORDER:
        spec = STAGE_SPECS[stage]
        should_run, reason = status[stage]
        
        if should_run:
            icon = "🔄"
            action = "RUN"
        else:
            icon = "⏭️ "
            action = "SKIP"
        
        print(f"  {icon} {action:4} {stage.value:12} - {reason}")
    print("-" * 60)


def parse_stage(stage_name: str) -> Stage:
    """Parse stage name string to Stage enum."""
    try:
        return Stage(stage_name.lower())
    except ValueError:
        valid = ", ".join(s.value for s in Stage)
        raise ValueError(f"Unknown stage '{stage_name}'. Valid stages: {valid}")


def clean_stale_artifacts(artifacts_dir: Path, verbose: bool = True) -> list[str]:
    """
    Remove stale artifacts from previous versions.
    
    Call this before running the pipeline to ensure clean state.
    
    Args:
        artifacts_dir: Path to artifacts directory
        verbose: If True, print what's being removed
        
    Returns:
        List of removed file/directory names
    """
    removed = []
    
    # Remove stale files
    for pattern in STALE_ARTIFACTS:
        path = artifacts_dir / pattern
        if path.exists():
            if verbose:
                print(f"[cleanup] Removing stale artifact: {pattern}")
            path.unlink()
            removed.append(pattern)
    
    # Remove stale directories
    for pattern in STALE_DIRECTORIES:
        path = artifacts_dir / pattern
        if path.exists() and path.is_dir():
            if verbose:
                print(f"[cleanup] Removing stale directory: {pattern}")
            shutil.rmtree(path)
            removed.append(pattern)
    
    return removed


def clean_all_artifacts(artifacts_dir: Path, verbose: bool = True) -> None:
    """
    Remove ALL artifacts for a fresh run.
    
    This is more aggressive than clean_stale_artifacts - it removes everything
    except the source video.
    
    Args:
        artifacts_dir: Path to artifacts directory
        verbose: If True, print what's being removed
    """
    if not artifacts_dir.exists():
        return
    
    # Get all expected outputs from all stages
    canonical_outputs = {
        "subtitles.srt",
        "subtitles.json", 
        "analysis.json",
        "story.md",
        "pages.json",
        "selected_frames.json",
        "frames",
        "cache",
    }
    
    for item in artifacts_dir.iterdir():
        if item.name.startswith("."):
            continue  # Skip hidden files
            
        if item.name in canonical_outputs or item.name in STALE_ARTIFACTS:
            if verbose:
                print(f"[cleanup] Removing: {item.name}")
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

