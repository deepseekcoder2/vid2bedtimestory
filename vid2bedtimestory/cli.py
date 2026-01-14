from __future__ import annotations

import json
import shutil
import time
import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import config


class PipelineTimer:
    """Track timing for pipeline stages."""
    
    def __init__(self):
        self.stages: dict[str, float] = {}
        self.start_time: float = None
        self.current_stage: str = None
        self.stage_start: float = None
    
    def start(self):
        """Start the overall pipeline timer."""
        self.start_time = time.time()
    
    def begin_stage(self, name: str):
        """Begin timing a stage."""
        self.current_stage = name
        self.stage_start = time.time()
    
    def end_stage(self) -> float:
        """End timing current stage, return elapsed seconds."""
        if self.stage_start is None:
            return 0.0
        elapsed = time.time() - self.stage_start
        self.stages[self.current_stage] = elapsed
        self.stage_start = None
        return elapsed
    
    def total_elapsed(self) -> float:
        """Get total elapsed time."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def print_summary(self, console: Console):
        """Print a timing summary table."""
        table = Table(title="⏱️  Pipeline Timing Summary")
        table.add_column("Stage", style="cyan")
        table.add_column("Time", justify="right", style="green")
        table.add_column("% of Total", justify="right", style="dim")
        
        total = self.total_elapsed()
        
        for stage, elapsed in self.stages.items():
            pct = (elapsed / total * 100) if total > 0 else 0
            table.add_row(
                stage,
                _format_duration(elapsed),
                f"{pct:.1f}%"
            )
        
        table.add_section()
        table.add_row("TOTAL", _format_duration(total), "100%", style="bold")
        
        console.print()
        console.print(table)


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


from .fallback import FallbackOptions, build_fallback_book
from .ffmpeg import extract_frame, extract_subtitles_to_srt, ffprobe_duration_s, pick_subtitle_stream
from .frame_selection import select_frames_for_book, FrameSelectionError
from .llm import write_story, paginate_story, enforce_alternating_layout, enforce_chronological_moments
from .models import AnalysisResult, BookSpec, SubtitleSegment
from .pdf import render_pdf, PDFWriteError
from .pipeline import Stage, STAGE_SPECS, should_run_stage, print_pipeline_plan, parse_stage, clean_stale_artifacts, clean_all_artifacts
from .moment_gap_fill import fill_moment_gaps
from .subtitles import parse_srt_file
from .validation import QualityValidator
from .version_check import check_system_requirements
from .video_analysis import analyze_video_v2
from .knowledge import load_franchise, list_available_franchises

app = typer.Typer()
console = Console()


@app.command()
def test():
    """Simple test command."""
    console.print("[green]Test command works![/green]")


@app.command()
def build(
    video: Path = typer.Argument(..., help="Path to video file with embedded subtitles"),
    out: Optional[Path] = typer.Option(None, help="Output PDF path. Defaults to out/<video_name>.pdf"),
    pages_target: int = typer.Option(config.default_pages_target, help="Target number of pages"),
    pages_min: int = typer.Option(config.default_pages_min, help="Minimum allowed pages"),
    pages_max: int = typer.Option(config.default_pages_max, help="Maximum allowed pages"),
    age_range: str = typer.Option(config.default_age_range, help="Target age range for content"),
    lang: str = typer.Option(config.default_lang, help="Preferred subtitle language"),
    artifacts_dir: Path = typer.Option(Path(config.artifacts_dir_default), help="Directory for intermediate artifacts"),
    cache_dir: Path = typer.Option(config.cache_dir, help="Directory for cached analysis results"),
    no_llm: bool = typer.Option(
        False,
        "--no-llm/--llm",
        help="Use fallback mode (no LLM) instead of MLX-VLM + MiMo pipeline.",
    ),
    keep_candidates: bool = typer.Option(
        False,
        "--keep-candidates",
        help="Keep VLM candidate frames for debugging.",
    ),
    rebuild_from: Optional[str] = typer.Option(
        None,
        "--rebuild-from",
        help="Force rebuild from this stage onwards (subtitles|analysis|story|gap_fill|pagination|screenshots|pdf|compress)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be run without actually running",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Clean all artifacts before running (fresh start)",
    ),
    franchise: str = typer.Option(
        ...,  # Required
        "--franchise",
        help="REQUIRED: Franchise ID for character database (e.g., 'hot_wheels_lets_race'). Use 'list' to see available.",
    ),
    srt: Optional[Path] = typer.Option(
        None,
        "--srt",
        help="Path to external SRT subtitle file. If provided, skips embedded subtitle extraction.",
    ),
    subtitle_context: int = typer.Option(
        50,
        "--subtitle-context",
        help="Number of subtitle lines used for gap detection. Increase for videos longer than 10 minutes.",
    ),
    ffprobe_path: Path = typer.Option(None, help="Path to ffprobe executable"),
    ffmpeg_path: Path = typer.Option(None, help="Path to ffmpeg executable"),
) -> None:
    """
    Build a children's picture book PDF from a video episode.
    
    Uses Makefile-style dependency tracking: stages are skipped if their
    outputs exist and are newer than their inputs. Use --rebuild-from to
    force regeneration from a specific stage.
    """
    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Error:[/red] Video file does not exist: {video}")
        raise typer.Exit(1)

    # Parse --rebuild-from if provided
    force_from: Optional[Stage] = None
    if rebuild_from:
        try:
            force_from = parse_stage(rebuild_from)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Set default output path if not provided
    if out is None:
        out = Path("out") / f"{video_path.stem}.pdf"
    else:
        out = Path(out) # Ensure it's a Path object

    # Apply config overrides
    if ffprobe_path:
        config.ffmpeg.ffprobe_path = ffprobe_path
    if ffmpeg_path:
        config.ffmpeg.ffmpeg_path = ffmpeg_path

    # Now check system requirements with the updated config
    try:
        check_system_requirements()
    except RuntimeError as e:
        console.print(f"[red]System requirement error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Processing video:[/green] {video}")
    console.print(f"[green]Output PDF:[/green] {out}")
    console.print(f"[green]Artifacts directory:[/green] {artifacts_dir}")
    console.print(f"[green]Target pages:[/green] {pages_target} (range: {pages_min}-{pages_max})")
    
    # Handle franchise database (REQUIRED)
    if franchise.lower() == "list":
        console.print("\n[bold]Available franchises:[/bold]")
        available = list_available_franchises()
        if available:
            for fid, fname, source in available:
                console.print(f"  • {fid} - {fname} [dim]({source})[/dim]")
        else:
            console.print("  [dim]No franchises found. Add JSON files to vid2bedtimestory/knowledge/franchises/[/dim]")
        raise typer.Exit(0)
    
    franchise_db = load_franchise(franchise)
    if not franchise_db:
        console.print(f"[red]Error:[/red] Franchise '{franchise}' not found.")
        console.print("Run with '--franchise list' to see available franchises.")
        console.print("Add new franchises to: vid2bedtimestory/knowledge/franchises/")
        raise typer.Exit(1)
    
    # Validate franchise has all required fields
    try:
        from .knowledge import validate_franchise
        validate_franchise(franchise_db)
    except Exception as e:
        console.print(f"[red]Error:[/red] Franchise validation failed:\n{e}")
        raise typer.Exit(1)
    
    console.print(f"[green]Franchise DB:[/green] {franchise_db.franchise_name} ({len(franchise_db.characters)} characters)")
    
    if franchise_db.pagination:
        pg = franchise_db.pagination
        pages_target = pg.target_pages
        pages_min = pg.min_pages
        pages_max = pg.max_pages
        console.print(f"[green]Pagination override:[/green] target={pages_target}, range={pages_min}-{pages_max}")

    # Create directories
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cleanup stale artifacts
    if fresh:
        console.print("[yellow]--fresh: Cleaning all artifacts for fresh start[/yellow]")
        clean_all_artifacts(artifacts_dir, verbose=True)
        # Also clean root cache (embedding index)
        if cache_dir.exists() and cache_dir != artifacts_dir:
            cache_file_count = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
            if cache_file_count > 0:
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"[yellow]  Removed {cache_file_count} files from {cache_dir}[/yellow]")
    else:
        removed = clean_stale_artifacts(artifacts_dir, verbose=False)
        if removed:
            console.print(f"[dim]Cleaned {len(removed)} stale artifact(s)[/dim]")

    # Show execution plan
    print_pipeline_plan(video_path, artifacts_dir, force_from)
    
    if dry_run:
        console.print("[yellow]Dry run - exiting without executing[/yellow]")
        return

    # Initialize pipeline timer
    timer = PipelineTimer()
    timer.start()

    try:
        # =====================================================================
        # STAGE: SUBTITLES
        # =====================================================================
        srt_path = artifacts_dir / "subtitles.srt"
        subtitles_json_path = artifacts_dir / "subtitles.json"
        
        timer.begin_stage("Subtitles")
        duration_s = ffprobe_duration_s(video_path)
        console.print(f"[dim]Video duration: {duration_s:.1f}s[/dim]")
        
        # Check if external SRT was provided
        if srt is not None:
            # External SRT mode - skip embedded extraction entirely
            external_srt_path = Path(srt)
            if not external_srt_path.exists():
                console.print(f"[red]Error:[/red] SRT file not found: {srt}")
                raise typer.Exit(1)
            
            console.print(f"[blue]Stage: SUBTITLES[/blue] (using external SRT: {external_srt_path.name})")
            
            # Copy external SRT to artifacts directory
            shutil.copy(external_srt_path, srt_path)
            
            # Parse the SRT file
            segments = parse_srt_file(srt_path)
            elapsed = timer.end_stage()
            console.print(f"[green]✓ Parsed {len(segments)} subtitle segments from external SRT[/green] [dim]({elapsed:.1f}s)[/dim]")
            
            # Save as JSON for downstream stages
            with open(subtitles_json_path, "w", encoding="utf-8") as f:
                json.dump([seg.model_dump() for seg in segments], f, indent=2, ensure_ascii=False)
        else:
            # Normal mode - extract from embedded subtitles
            run_subtitles, reason = should_run_stage(Stage.SUBTITLES, video_path, artifacts_dir, force_from)
            
            if run_subtitles:
                console.print(f"[blue]Stage: SUBTITLES[/blue] ({reason})")
                
                subtitle_stream = pick_subtitle_stream(video_path, preferred_languages=[lang])
                console.print(f"[dim]Found subtitle stream: index={subtitle_stream.index}, lang={subtitle_stream.language}[/dim]")
                
                extract_subtitles_to_srt(video_path, srt_path, subtitle_stream.index)
                segments = parse_srt_file(srt_path)
                elapsed = timer.end_stage()
                console.print(f"[green]✓ Parsed {len(segments)} subtitle segments[/green] [dim]({elapsed:.1f}s)[/dim]")
                
                with open(subtitles_json_path, "w", encoding="utf-8") as f:
                    json.dump([seg.model_dump() for seg in segments], f, indent=2, ensure_ascii=False)
            else:
                console.print(f"[yellow]Stage: SUBTITLES[/yellow] skipped ({reason})")
                with open(subtitles_json_path, "r", encoding="utf-8") as f:
                    segments = [SubtitleSegment.model_validate(s) for s in json.load(f)]
                timer.end_stage()
        
        subtitles_text = " ".join(seg.text for seg in segments)

        if no_llm:
            # =====================================================================
            # FALLBACK MODE (no LLM)
            # =====================================================================
            timer.begin_stage("Fallback")
            console.print("[blue]Using fallback mode (no LLM)...[/blue]")
            opts = FallbackOptions(target_pages=pages_target)
            book_spec = build_fallback_book(
                title=f"Episode from {video_path.stem}",
                segments=segments,
                duration_s=duration_s,
                opts=opts,
            )
            analysis_result = None
            timer.end_stage()
        else:
            # =====================================================================
            # STAGE: ANALYSIS
            # =====================================================================
            run_analysis, reason = should_run_stage(Stage.ANALYSIS, video_path, artifacts_dir, force_from)
            analysis_json_path = artifacts_dir / "analysis.json"
            
            timer.begin_stage("Analysis")
            if run_analysis:
                console.print(f"[blue]Stage: ANALYSIS[/blue] ({reason})")
                analysis_result = analyze_video_v2(
                    video_path, 
                    segments, 
                    duration_s,
                    franchise_db=franchise_db,
                    subtitle_context_limit=subtitle_context,
                )
                with open(analysis_json_path, "w", encoding="utf-8") as f:
                    json.dump(analysis_result.model_dump(), f, indent=2, ensure_ascii=False)
                elapsed = timer.end_stage()
                console.print(f"[green]✓ Analysis complete: {len(analysis_result.moments)} moments, {len(analysis_result.characters)} characters[/green] [dim]({_format_duration(elapsed)})[/dim]")
            else:
                console.print(f"[yellow]Stage: ANALYSIS[/yellow] skipped ({reason})")
                with open(analysis_json_path, "r", encoding="utf-8") as f:
                    analysis_result = AnalysisResult.model_validate(json.load(f))
                timer.end_stage()

            # =====================================================================
            # STAGE: STORY
            # =====================================================================
            run_story, reason = should_run_stage(Stage.STORY, video_path, artifacts_dir, force_from)
            story_md_path = artifacts_dir / "story.md"
            
            timer.begin_stage("Story")
            if run_story:
                console.print(f"[blue]Stage: STORY[/blue] ({reason})")
                story_text = write_story(
                    analysis_result, 
                    subtitles_text, 
                    franchise_db=franchise_db,
                    target_pages=pages_target,
                )
                with open(story_md_path, "w", encoding="utf-8") as f:
                    f.write(story_text)
                elapsed = timer.end_stage()
                console.print(f"[green]✓ Story written: {len(story_text)} characters[/green] [dim]({_format_duration(elapsed)})[/dim]")
            else:
                console.print(f"[yellow]Stage: STORY[/yellow] skipped ({reason})")
                with open(story_md_path, "r", encoding="utf-8") as f:
                    story_text = f.read()
                timer.end_stage()

            # =====================================================================
            # STAGE: GAP_FILL (VideoAgent-style iterative retrieval)
            # =====================================================================
            run_gap_fill, reason = should_run_stage(Stage.GAP_FILL, video_path, artifacts_dir, force_from)
            analysis_enriched_path = artifacts_dir / "analysis_enriched.json"
            
            timer.begin_stage("Gap Fill")
            if run_gap_fill:
                console.print(f"[blue]Stage: GAP_FILL[/blue] ({reason})")
                
                analysis_result, gap_result = fill_moment_gaps(
                    story_text=story_text,
                    analysis=analysis_result,
                    subtitles=segments,
                    video_path=video_path,
                    cache_dir=cache_dir,
                    franchise_db=franchise_db,
                    max_retrievals=10,  # Cap to stay frame-efficient
                )
                
                # Save enriched analysis
                with open(analysis_enriched_path, "w", encoding="utf-8") as f:
                    json.dump(analysis_result.model_dump(), f, indent=2, ensure_ascii=False)
                
                elapsed = timer.end_stage()
                console.print(
                    f"[green]✓ Gap fill: {gap_result.gaps_filled}/{gap_result.gaps_found} gaps filled[/green] "
                    f"[dim]({_format_duration(elapsed)})[/dim]"
                )
            else:
                console.print(f"[yellow]Stage: GAP_FILL[/yellow] skipped ({reason})")
                # Load enriched analysis if it exists, otherwise use original
                if analysis_enriched_path.exists():
                    with open(analysis_enriched_path, "r", encoding="utf-8") as f:
                        analysis_result = AnalysisResult.model_validate(json.load(f))
                timer.end_stage()

            # =====================================================================
            # STAGE: PAGINATION
            # =====================================================================
            run_pagination, reason = should_run_stage(Stage.PAGINATION, video_path, artifacts_dir, force_from)
            pages_json_path = artifacts_dir / "pages.json"
            
            timer.begin_stage("Pagination")
            if run_pagination:
                console.print(f"[blue]Stage: PAGINATION[/blue] ({reason})")
                words_per_page = 45
                if franchise_db and franchise_db.pagination:
                    words_per_page = franchise_db.pagination.words_per_page_target
                
                book_spec = paginate_story(
                    story_text, 
                    analysis_result,
                    franchise_db=franchise_db,
                    target_pages=pages_target,
                    min_pages=pages_min,
                    max_pages=pages_max,
                    words_per_page=words_per_page,
                )
                
                # Enforce deterministic alternating layout
                book_spec = enforce_alternating_layout(book_spec)
                
                # Enforce chronological moment ordering
                book_spec = enforce_chronological_moments(book_spec, analysis_result)
                
                with open(pages_json_path, "w", encoding="utf-8") as f:
                    json.dump(book_spec.model_dump(), f, indent=2, ensure_ascii=False)
                elapsed = timer.end_stage()
                console.print(f"[green]✓ Paginated into {len(book_spec.pages)} pages[/green] [dim]({_format_duration(elapsed)})[/dim]")
            else:
                console.print(f"[yellow]Stage: PAGINATION[/yellow] skipped ({reason})")
                with open(pages_json_path, "r", encoding="utf-8") as f:
                    book_spec = BookSpec.model_validate(json.load(f))
                timer.end_stage()

        # Page count check
        page_count = len(book_spec.pages)
        if page_count < pages_min or page_count > pages_max:
            console.print(
                f"[yellow]WARNING:[/yellow] Generated {page_count} pages "
                f"(target range: {pages_min}-{pages_max})"
            )

        # =====================================================================
        # STAGE: SCREENSHOTS
        # =====================================================================
        frames_dir = artifacts_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        timer.begin_stage("Screenshots")
        if not no_llm:
            run_screenshots, reason = should_run_stage(Stage.SCREENSHOTS, video_path, artifacts_dir, force_from)
            
            if run_screenshots:
                console.print(f"[blue]Stage: SCREENSHOTS[/blue] ({reason})")
                
                # New clean frame selection
                selected_frames = select_frames_for_book(
                    pages=book_spec.pages,
                    video_path=video_path,
                    subtitles=segments,
                    analysis=analysis_result,
                    frames_dir=frames_dir,
                    temp_dir=cache_dir / "segments",
                    video_duration=duration_s,
                )
                
                # Update pages with selected timestamps
                selected_by_page = {sf.page_index: sf for sf in selected_frames}
                for page in book_spec.pages:
                    sf = selected_by_page.get(page.page_index)
                    if sf:
                        page.image_timestamp_candidates_s = [sf.timestamp_s]
                
                # Save selection decisions
                selected_json_path = artifacts_dir / "selected_frames.json"
                with open(selected_json_path, "w", encoding="utf-8") as f:
                    json.dump([sf.model_dump() for sf in selected_frames], f, indent=2, ensure_ascii=False)
                
                # Update pages.json with final timestamps
                pages_json_path = artifacts_dir / "pages.json"
                with open(pages_json_path, "w", encoding="utf-8") as f:
                    json.dump(book_spec.model_dump(), f, indent=2, ensure_ascii=False)
                
                elapsed = timer.end_stage()
                console.print(f"[green]✓ Selected {len(selected_frames)} screenshots[/green] [dim]({_format_duration(elapsed)})[/dim]")
            else:
                console.print(f"[yellow]Stage: SCREENSHOTS[/yellow] skipped ({reason})")
                timer.end_stage()
        else:
            # Fallback: extract frames at proportional timestamps
            for page in book_spec.pages:
                if page.image_timestamp_candidates_s:
                    timestamp_s = page.image_timestamp_candidates_s[0]
                    frame_path = frames_dir / f"page_{page.page_index:03d}.png"
                    if not frame_path.exists():
                        extract_frame(video_path, timestamp_s, frame_path)
            timer.end_stage()

        # =====================================================================
        # STAGE: PDF
        # =====================================================================
        timer.begin_stage("PDF")
        run_pdf, reason = should_run_stage(Stage.PDF, video_path, artifacts_dir, force_from)
        console.print(f"[blue]Stage: PDF[/blue] ({reason})")
        
        # Validate quality gates
        validator = QualityValidator(
            min_pages=pages_min,
            max_pages=pages_max,
            max_sentence_length=50,
        )
        issues = validator.validate_book_spec(book_spec, frames_dir)
        if issues:
            console.print(f"[yellow]Quality warnings ({len(issues)}):[/yellow]")
            for issue in issues[:5]:
                console.print(f"  [dim]- {issue}[/dim]")
            if len(issues) > 5:
                console.print(f"  [dim]... and {len(issues) - 5} more[/dim]")
        
        try:
            render_pdf(book_spec, frames_dir, out)
        except PDFWriteError as e:
            elapsed = timer.end_stage()
            console.print(f"[red]✗ PDF write failed:[/red] {e}")
            console.print("[yellow]Hint:[/yellow] If you have the PDF open in Preview, close it and run again.")
            raise typer.Exit(1)
        
        elapsed = timer.end_stage()
        console.print(f"[green]✓ PDF rendered to {out}[/green] [dim]({_format_duration(elapsed)})[/dim]")

        # =====================================================================
        # STAGE: COMPRESS
        # =====================================================================
        timer.begin_stage("Compress")
        run_compress, reason = should_run_stage(Stage.COMPRESS, video_path, artifacts_dir, force_from)
        
        if run_compress:
            compressed_out = out.with_name(f"{out.stem}-compress.pdf")
            console.print(f"[blue]Stage: COMPRESS[/blue] ({reason})")
            
            # Call the standalone script using the current python interpreter
            cmd = [sys.executable, "scripts/compress_pdf.py", str(out), str(compressed_out)]
            
            try:
                subprocess.run(cmd, check=True)
                elapsed = timer.end_stage()
                console.print(f"[green]✓ Compressed PDF created: {compressed_out}[/green] [dim]({elapsed:.1f}s)[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Compression failed but PDF is intact: {e}[/yellow]")
                timer.end_stage()
        else:
            console.print(f"[yellow]Stage: COMPRESS[/yellow] skipped ({reason})")
            timer.end_stage()

        # Cleanup
        if not keep_candidates:
            candidates_dir = frames_dir / "_candidates"
            if candidates_dir.exists():
                candidate_count = sum(1 for _ in candidates_dir.rglob("*.png"))
                shutil.rmtree(candidates_dir)
                console.print(f"[dim]Cleaned up {candidate_count} candidate frames[/dim]")

        # Print timing summary
        timer.print_summary(console)
        
        console.print(f"\n[green]Success![/green] Generated {page_count}-page book: {out}")

    except FrameSelectionError as e:
        console.print(f"[red]Frame selection failed:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def status(
    video: Path = typer.Argument(..., help="Path to video file"),
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Directory containing artifacts"),
) -> None:
    """
    Show pipeline status for a video - what stages would run/skip.
    """
    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Error:[/red] Video file does not exist: {video}")
        raise typer.Exit(1)
    
    print_pipeline_plan(video_path, artifacts_dir)


@app.command()
def clean(
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Directory containing artifacts"),
    cache_only: bool = typer.Option(False, "--cache-only", help="Only clear cache, keep analysis artifacts"),
    all_artifacts: bool = typer.Option(False, "--all", help="Remove entire artifacts directory"),
) -> None:
    """
    Clean up temporary files and caches from previous runs.
    """
    if not artifacts_dir.exists():
        console.print(f"[yellow]Nothing to clean:[/yellow] {artifacts_dir} does not exist")
        return
    
    cleaned = 0
    
    if all_artifacts:
        file_count = sum(1 for _ in artifacts_dir.rglob("*") if _.is_file())
        shutil.rmtree(artifacts_dir)
        console.print(f"[green]Removed entire artifacts directory:[/green] {file_count} files deleted")
        return
    
    for candidates_dir in artifacts_dir.rglob("_candidates"):
        if candidates_dir.is_dir():
            count = sum(1 for _ in candidates_dir.rglob("*.png"))
            shutil.rmtree(candidates_dir)
            cleaned += count
            console.print(f"[dim]Removed {count} candidate frames[/dim]")
    
    cache_dir = artifacts_dir / "cache"
    if cache_dir.exists():
        cache_count = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
        shutil.rmtree(cache_dir)
        cleaned += cache_count
        console.print(f"[dim]Removed {cache_count} cached files[/dim]")
    
    if cleaned > 0:
        console.print(f"[green]Cleanup complete:[/green] {cleaned} files removed")
    else:
        console.print("[yellow]No temporary files to clean[/yellow]")


@app.command()
def render(
    pages: Path = typer.Option(..., help="Path to pages.json file"),
    frames_dir: Path = typer.Option(..., help="Directory containing extracted frames"),
    out: Path = typer.Option(Path("out/book.pdf"), help="Output PDF path"),
) -> None:
    """
    Render a PDF from an existing pages.json and frames directory.
    """
    if not pages.exists():
        console.print(f"[red]Error:[/red] pages.json not found: {pages}")
        raise typer.Exit(1)
    
    if not frames_dir.exists():
        console.print(f"[red]Error:[/red] Frames directory not found: {frames_dir}")
        raise typer.Exit(1)

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pages, "r", encoding="utf-8") as f:
            book_spec = BookSpec.model_validate(json.load(f))

        render_pdf(book_spec, frames_dir, out)
        console.print(f"[green]Success![/green] PDF rendered to {out}")

    except PDFWriteError as e:
        console.print(f"[red]PDF write failed:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
