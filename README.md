# Bookify — Video → Children's Picture Book

Convert video episodes into **print-ready A4 PDF** children's picture books (ages 5–8, ~22 pages).

## Quick Start

```bash
# Setup
./setup_mac.sh

# Add API key
echo "your-openrouter-key" > openrouterapikey.md

# Run with franchise database (recommended for known shows)
python -m bookify build videos/episode.mkv --llm --franchise hot_wheels_lets_race

# Run without franchise (discovery mode)
python -m bookify build videos/episode.mkv --llm

# List available franchises
python -m bookify build videos/episode.mkv --franchise list
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BOOKIFY PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  VIDEO FILE (.mkv)                                                       │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ SUBTITLES   │ Extract embedded subtitles                              │
│  │ Stage       │ → subtitles.srt, subtitles.json                         │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐     ┌──────────────────┐                               │
│  │ ANALYSIS    │ ←── │ Franchise DB     │ (optional)                    │
│  │ Stage       │     │ hot_wheels.json  │                               │
│  └──────┬──────┘     └──────────────────┘                               │
│         │                                                                │
│         │  VideoAgent Pipeline:                                          │
│         │  1. Sparse Survey (12 frames)                                  │
│         │  2. Beat Detection (LLM)                                       │
│         │  3. Deep Dive (VLM per beat)                                   │
│         │  4. Character Extraction (DB or multi-frame consensus)         │
│         │  5. Assembly                                                   │
│         │                                                                │
│         ▼                                                                │
│  → analysis.json (characters, beats, moments)                            │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐                                                         │
│  │ STORY       │ LLM writes children's story                             │
│  │ Stage       │ Uses: pronouns, catchphrases, story beats               │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         ▼                                                                │
│  → story.md                                                              │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐                                                         │
│  │ PAGINATION  │ LLM splits into ~22 pages                               │
│  │ Stage       │ Matches text to timestamps                              │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         ▼                                                                │
│  → pages.json                                                            │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐                                                         │
│  │ SCREENSHOTS │ VLM scores candidate frames per page                    │
│  │ Stage       │ Cloud VLM (60 concurrent) or local MLX-VLM              │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         ▼                                                                │
│  → frames/page_001.png ... page_022.png                                  │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐                                                         │
│  │ PDF         │ Render final book                                       │
│  │ Stage       │ A4, full-bleed images, typeset text                     │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         ▼                                                                │
│  → out/book.pdf                                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Stages & Dependencies

| Stage | Inputs | Outputs | Description |
|-------|--------|---------|-------------|
| **SUBTITLES** | `video.mkv` | `subtitles.srt`, `subtitles.json` | Extract embedded subtitles |
| **ANALYSIS** | `video.mkv`, `subtitles.json` | `analysis.json` | VideoAgent pipeline + character extraction |
| **STORY** | `analysis.json`, `subtitles.json` | `story.md` | LLM writes children's narrative |
| **PAGINATION** | `story.md`, `analysis.json` | `pages.json` | Split into pages with timestamps |
| **SCREENSHOTS** | `video.mkv`, `pages.json`, `analysis.json` | `frames/`, `selected_frames.json` | VLM selects best frame per page |
| **PDF** | `pages.json`, `frames/` | `out/book.pdf` | Render final PDF |

**Makefile-style execution**: Stages only run if outputs are missing or inputs are newer.

## Character Knowledge Base

For known franchises, use a character database for **100% accurate pronouns**:

```bash
python -m bookify build video.mkv --llm --franchise hot_wheels_lets_race
```

### How It Works

| Mode | Character Source | Pronoun Accuracy |
|------|-----------------|------------------|
| **With franchise DB** | Database lookup + alias/fuzzy matching | ✅ 100% |
| **Without franchise** | Signal analysis + multi-frame VLM consensus | ~85-95% |

### Franchise Database Features

```json
{
  "characters": {
    "coop": {
      "display_name": "Coop",
      "aliases": ["Coop", "the new kid"],
      "pronoun": "he/him",
      "role": "protagonist",
      "visual_signature": { "hair": "spiky two-tone", "clothing": "teal hoodie" },
      "catchphrases": ["Challenge accepted!"],
      "relationships": { "dash_wheeler": "mentor" }
    }
  },
  "known_non_characters": ["campers", "everyone", "crowd"]
}
```

**Benefits:**
- Correct pronouns (no VLM guessing)
- Catchphrases injected into story
- Excludes false positives ("Campers" won't become a character)
- Fuzzy matching handles typos

### Adding a New Franchise

1. Create `bookify/knowledge/franchises/your_show.json`
2. Follow the schema in `hot_wheels_lets_race.json`
3. Use with `--franchise your_show`

Or add to user config: `~/.bookify/franchises/your_show.json`

## Character Detection (Discovery Mode)

When no franchise DB is provided, the pipeline uses **signal analysis**:

```
Subtitle name "Coop"
    │
    ▼
Signal Analysis:
  - dialogue_count: 12 (speaks often)
  - individual_actions: 8 ("Coop grabbed...")
  - plural_usage: 0
    │
    ▼
Score: 0.95 → HIGH CONFIDENCE CHARACTER
    │
    ▼
Multi-frame VLM consensus (3 frames)
    │
    ▼
Character profile with pronoun
```

**Signals analyzed:**
- Dialogue attribution (`[Coop] Challenge accepted!`)
- Individual actions (`Coop grabbed the wheel`)
- Plural/collective usage (`Hey campers!`)
- Frequency of mentions

## CLI Reference

```bash
python -m bookify build VIDEO [OPTIONS]
```

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--out PATH` | `out/book.pdf` | Output PDF path |
| `--pages-target N` | `22` | Target page count |
| `--pages-min N` | `18` | Minimum pages |
| `--pages-max N` | `40` | Maximum pages |
| `--llm / --no-llm` | `--no-llm` | Enable full LLM pipeline |

### Franchise & Character

| Option | Description |
|--------|-------------|
| `--franchise ID` | Use character database (e.g., `hot_wheels_lets_race`) |
| `--franchise list` | List available franchise databases |

### Pipeline Control

| Option | Description |
|--------|-------------|
| `--rebuild-from STAGE` | Force rebuild from stage onwards |
| `--fresh` | Clean all artifacts before running |
| `--dry-run` | Show execution plan without running |

### VLM Backend

| Option | Description |
|--------|-------------|
| `--vlm cloud` | Use OpenRouter cloud VLM (default, fast) |
| `--vlm local` | Use local MLX-VLM (slower, free) |

### Debugging

| Option | Description |
|--------|-------------|
| `--keep-candidates` | Keep VLM candidate frames (~300MB) |
| `--artifacts-dir PATH` | Custom artifacts directory |

## Output Files

| File | Description |
|------|-------------|
| `out/book.pdf` | Final print-ready A4 PDF |
| `artifacts/subtitles.srt` | Extracted subtitles |
| `artifacts/subtitles.json` | Parsed subtitle segments |
| `artifacts/analysis.json` | Video analysis (characters, beats, moments) |
| `artifacts/story.md` | Generated story text |
| `artifacts/pages.json` | Page structure with timestamps |
| `artifacts/frames/` | Selected screenshot frames |
| `artifacts/selected_frames.json` | Frame selection metadata |

## Prerequisites

- **Mac**: Apple Silicon (M2/M3/M4) with 16GB+ RAM
- **Python**: 3.10+
- **FFmpeg**: `brew install ffmpeg`
- **OpenRouter API Key**: [openrouter.ai](https://openrouter.ai)

Optional (for local VLM):
- **MLX-VLM**: `pip install mlx-vlm`
- **32GB+ RAM** recommended

## Project Structure

```
nathanbook/
├── bookify/
│   ├── cli.py                 # CLI entry point
│   ├── config.py              # Model configuration
│   ├── pipeline.py            # Stage dependency management
│   ├── prompts.py             # LLM prompts (story, pagination)
│   ├── llm.py                 # Story writing, pagination
│   ├── vlm.py                 # VLM scoring (cloud + local)
│   ├── models.py              # Pydantic schemas
│   ├── ffmpeg.py              # FFmpeg wrappers
│   ├── pdf.py                 # PDF rendering
│   ├── screenshot_selection.py # VLM-based frame selection
│   ├── knowledge/             # Character databases
│   │   ├── loader.py          # Franchise DB loader
│   │   └── franchises/        # Franchise JSON files
│   │       └── hot_wheels_lets_race.json
│   └── video_analysis/        # VideoAgent pipeline
│       ├── __init__.py        # analyze_video_v2()
│       ├── character_signals.py # Signal analysis
│       ├── types.py           # Internal types
│       ├── prompts.py         # VLM prompts
│       └── phases/            # Pipeline phases
│           ├── sparse_survey.py
│           ├── beat_detection.py
│           ├── deep_dive.py
│           └── character_extraction.py
├── videos/                    # Source videos
├── artifacts/                 # Generated intermediates
├── out/                       # Generated PDFs
└── reference/                 # PRD, documentation
```

## Configuration

Edit `bookify/config.py`:
- `creative_model`: OpenRouter model for story/pagination
- `vlm_cloud_model`: Cloud VLM for frame scoring
- `vlm_cloud_max_concurrent`: Parallel requests (default: 60)
- `vlm_min_score_threshold`: Minimum VLM score for frame selection (default: 5.0)

API key location: `openrouterapikey.md` (first non-comment line)

## Examples

```bash
# Full pipeline with Hot Wheels character database
python -m bookify build videos/episode101-001.mkv --llm --franchise hot_wheels_lets_race

# Fresh run (clean all artifacts first)
python -m bookify build videos/episode.mkv --llm --fresh

# Rebuild from story stage (reuse analysis)
python -m bookify build videos/episode.mkv --llm --rebuild-from story

# Check what would run without executing
python -m bookify build videos/episode.mkv --llm --dry-run

# Use local VLM instead of cloud
python -m bookify build videos/episode.mkv --llm --vlm local
```

## Performance

| Stage | Time (typical) | Notes |
|-------|----------------|-------|
| Subtitles | ~5s | FFmpeg extraction |
| Analysis | ~3-5 min | VLM calls for frames |
| Story | ~30s | Single LLM call |
| Pagination | ~30s | Single LLM call |
| Screenshots | ~8-12 min | 41 frames × 20 pages with cloud VLM |
| PDF | ~10s | Rendering |
| **Total** | ~15-20 min | With cloud VLM |
