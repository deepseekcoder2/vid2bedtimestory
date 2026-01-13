# Vid2BedtimeStory

Convert children's TV episodes into **print-ready picture book PDFs**.

Takes a video file with subtitles → outputs a ~30-page illustrated storybook for ages 5-8.

---

## Requirements Overview

This tool uses a hybrid local + cloud AI architecture:

| Component | What It Does | Where It Runs |
|-----------|--------------|---------------|
| **Video Analysis** | Watches video, extracts scenes/characters | 🖥️ Local (32B VLM) |
| **Frame Embedding** | Indexes frames for semantic search | 🖥️ Local (2B model) |
| **Frame Reranking** | Picks best frame for each page | 🖥️ Local (2B model) |
| **Story Writing** | Writes the children's story | ☁️ Cloud (Claude) |
| **Frame Scoring** | Scores candidate frames | ☁️ Cloud (Qwen VL) |

**Why local + cloud?** Cloud APIs can't accept 500MB+ video uploads. The heavy video analysis runs locally using Apple's MLX framework, while text generation uses cloud APIs.

---

## System Requirements

| Requirement | Details |
|-------------|---------|
| **Mac with Apple Silicon** | M1/M2/M3/M4 with **32GB+ RAM** (64GB recommended) |
| **Python 3.10+** | `python3 --version` to check |
| **FFmpeg** | `brew install ffmpeg` |
| **Ghostscript** | `brew install ghostscript` (for PDF compression) |
| **~30GB disk space** | For AI model downloads |
| **OpenRouter API Key** | [openrouter.ai](https://openrouter.ai) |

### Platform Support

| Platform | Status |
|----------|--------|
| **Mac (Apple Silicon, 32GB+ RAM)** | ✅ Full support |
| **Mac (Apple Silicon, 16GB RAM)** | ⚠️ May work with swap, will be slow |
| **Mac (Intel)** | ❌ Not supported (no MLX) |
| **Windows / Linux** | ❌ Requires code modification (replace MLX workers with PyTorch) |

---

## AI Models Required

### Local Models (Auto-downloaded on first run)

These models download automatically from HuggingFace when you first run the pipeline:

| Model | Size | Purpose | Framework |
|-------|------|---------|-----------|
| **Qwen3-VL-32B-Instruct-8bit** | ~18GB | Video analysis, scene extraction | MLX |
| **Qwen3-VL-Embedding-2B** | ~4GB | Frame semantic search | PyTorch |
| **Qwen3-VL-Reranker-2B** | ~4GB | Frame selection refinement | PyTorch |

**Total download: ~26GB** (stored in `~/.cache/huggingface/hub/`)

#### Pre-downloading Models (Optional)

If you want to download models before running the pipeline:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Download the main video analysis model (MLX)
python -c "from mlx_vlm import load; load('mlx-community/Qwen3-VL-32B-Instruct-8bit')"

# Download embedding model (PyTorch)
python -c "from transformers import AutoModel, AutoProcessor; AutoModel.from_pretrained('Qwen/Qwen3-VL-Embedding-2B', trust_remote_code=True); AutoProcessor.from_pretrained('Qwen/Qwen3-VL-Embedding-2B', trust_remote_code=True)"

# Download reranker model (PyTorch)
python -c "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-Reranker-2B', trust_remote_code=True); AutoProcessor.from_pretrained('Qwen/Qwen3-VL-Reranker-2B', trust_remote_code=True)"
```

### Cloud APIs (via OpenRouter)

These models run in the cloud via [OpenRouter](https://openrouter.ai):

| Model | Purpose |
|-------|---------|
| **Claude Sonnet 4.5** | Story writing, pagination |
| **Qwen3-VL-235B** | Frame scoring |
| **GPT-4o-mini** | Utility tasks |

See [OpenRouter pricing](https://openrouter.ai/models) for current rates.

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/vid2bedtimestory/vid2bedtimestory.git
cd vid2bedtimestory
./setup_mac.sh
```

### 2. Activate Environment

```bash
source .venv/bin/activate
```

### 3. Add Your OpenRouter API Key

Create an account at [openrouter.ai](https://openrouter.ai), add credits, and generate an API key.

```bash
echo "sk-or-v1-your-key-here" > openrouterapikey.md
```

### 4. Run It

```bash
python -m vid2bedtimestory build videos/episode.mkv --franchise hot_wheels_lets_race
```

**Note:** First run will be slower as models download (~26GB). Subsequent runs use cached models.

---

## Preparing Your Video

### Option A: Video with Embedded Subtitles (Recommended)

Your video file must have subtitles embedded (common in `.mkv` files). The tool extracts them automatically.

```bash
# Check if your video has subtitles
ffprobe -v error -select_streams s -show_entries stream=index,codec_name:stream_tags=language -of csv=p=0 your_video.mkv
```

### Option B: Separate SRT File

If your video doesn't have embedded subtitles, provide an external `.srt` file:

```bash
python -m vid2bedtimestory build videos/episode.mp4 --srt subtitles/episode.srt --franchise hot_wheels_lets_race
```

The SRT file must be timed to match the video.

---

## The Franchise File (Required)

Every run requires a `--franchise` flag pointing to a character database. This ensures:
- Correct character names and pronouns
- Proper catchphrases in the story
- Accurate character identification

### Using an Existing Franchise

```bash
# List available franchises
python -m vid2bedtimestory build video.mkv --franchise list

# Use Hot Wheels (included)
python -m vid2bedtimestory build video.mkv --franchise hot_wheels_lets_race
```

### Creating a New Franchise File

Create `vid2bedtimestory/knowledge/franchises/your_show.json`:

```json
{
  "franchise_id": "your_show",
  "franchise_name": "Your Show Name",
  
  "characters": {
    "main_character": {
      "display_name": "Character Name",
      "aliases": ["Nickname", "Other Name"],
      "pronoun": "she/her",
      "role": "protagonist",
      "traits": ["brave", "curious"],
      "catchphrases": ["Let's do this!"],
      "visual_signature": {
        "hair": "long brown hair",
        "clothing": "red jacket"
      }
    }
  },
  
  "known_non_characters": ["everyone", "crowd", "people"],
  
  "visual_style": {
    "shot_preferences": ["Action shots with characters visible"],
    "avoid": ["Extreme close-ups", "Dark scenes"]
  },
  
  "pagination": {
    "target_pages": 30,
    "min_pages": 26,
    "max_pages": 34,
    "words_per_page_target": 40
  },

  "prompt_examples": {
    "video_analysis": {
      "visual_description": {
        "good": "Character runs through forest, arms pumping, leaves flying past. Sunlight filters through trees.",
        "bad": "Character runs."
      }
    },
    "story_writing": {
      "opening": "Example opening paragraph for your show's style...",
      "climax": "Example climax paragraph..."
    }
  }
}
```

See `vid2bedtimestory/knowledge/franchises/hot_wheels_lets_race.json` for a complete example.

---

## Pipeline Stages

The tool runs through these stages (each can be cached/resumed):

| Stage | What Happens |
|-------|--------------|
| 1. **Subtitles** | Extract dialogue from video |
| 2. **Video Analysis** | Local AI scans video for scenes/characters |
| 3. **Story Writing** | Cloud AI writes children's story |
| 4. **Pagination** | Split story into pages |
| 5. **Frame Selection** | AI picks best frame per page |
| 6. **PDF Generation** | Render the final book |

Use `--rebuild-from <stage>` to resume from a specific stage.

---

## Common Commands

```bash
# Basic run
python -m vid2bedtimestory build video.mkv --franchise your_franchise

# Fresh start (clear all cached work)
python -m vid2bedtimestory build video.mkv --franchise your_franchise --fresh

# Rebuild from a specific stage (reuse earlier work)
python -m vid2bedtimestory build video.mkv --franchise your_franchise --rebuild-from story

# See what would run without running it
python -m vid2bedtimestory build video.mkv --franchise your_franchise --dry-run

# Use external subtitles
python -m vid2bedtimestory build video.mp4 --srt subs.srt --franchise your_franchise
```

---

## Output

After a successful run, you'll find:

| File | Description |
|------|-------------|
| `out/<video_name>.pdf` | Your picture book! |
| `out/<video_name>-compress.pdf` | Smaller version for sharing |
| `artifacts/story.md` | Generated story text |
| `artifacts/frames/` | Selected screenshots |

---

## Options Reference

| Option | Description |
|--------|-------------|
| `--franchise ID` | **Required.** Character database to use |
| `--srt PATH` | External SRT subtitle file |
| `--out PATH` | Output PDF path (default: `out/<video>.pdf`) |
| `--pages-target N` | Target page count (default: from franchise) |
| `--subtitle-context N` | Subtitle lines for gap detection (default: 50) |
| `--rebuild-from STAGE` | Force rebuild from: `subtitles`, `analysis`, `story`, `pagination`, `screenshots`, `pdf` |
| `--fresh` | Delete all artifacts and start over |
| `--dry-run` | Show plan without executing |
| `--keep-candidates` | Keep candidate frames for debugging |
| `--artifacts-dir PATH` | Custom artifacts directory |

### Note on Video Length

This tool was tested primarily on **10-11 minute** episodes. For longer episodes:

```bash
# For a 20-minute episode
python -m vid2bedtimestory build long_episode.mkv --franchise your_show --subtitle-context 100

# For a 30-minute episode  
python -m vid2bedtimestory build long_episode.mkv --franchise your_show --subtitle-context 150
```

Rough guideline: ~5 lines per minute of video.

---

## Troubleshooting

### "No subtitle stream found"
Your video doesn't have embedded subtitles. Use `--srt` with an external subtitle file.

### "Franchise not found"
Run `--franchise list` to see available options, or create your own JSON file.

### "openrouterapikey.md not found"
```bash
echo "sk-or-v1-your-key-here" > openrouterapikey.md
```

### Model download seems stuck
Large models (~18GB) can take a while. Check your internet connection and disk space.

### Out of memory / System becomes unresponsive
- Close other applications (browsers, IDEs)
- The 32B model needs ~24GB during inference
- If you only have 32GB RAM, enable macOS memory pressure handling
- 64GB RAM is recommended for smooth operation

### "Metal initialization" or GPU errors
The MLX workers run in separate processes to avoid conflicts. If you see Metal errors:
```bash
# Clear any cached state and try again
python -m vid2bedtimestory build video.mkv --franchise your_show --fresh
```

### Pipeline seems stuck
Video analysis and frame selection are the longest stages. They process the full video and score many candidate frames. Be patient, especially on first run.

### PyTorch/Transformers errors
Make sure you installed all dependencies:
```bash
source .venv/bin/activate
pip install torch transformers
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Video                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Video Analysis (LOCAL - MLX)                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Qwen3-VL-32B-Instruct-8bit                              ││
│  │ • Watches full video at 8fps                            ││
│  │ • Extracts scenes, characters, story beats              ││
│  │ • Outputs structured JSON analysis                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2-4: Story Generation (CLOUD - OpenRouter)            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Claude Sonnet 4.5                                       ││
│  │ • Writes children's story from analysis                 ││
│  │ • Paginates into ~30 pages                              ││
│  │ • Generates visual targets for each page                ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Frame Selection (LOCAL + CLOUD)                    │
│  ┌──────────────────────┐  ┌──────────────────────────────┐│
│  │ Embedding (LOCAL)    │  │ Scoring (CLOUD)              ││
│  │ Qwen3-VL-Embedding-2B│  │ Qwen3-VL-235B                ││
│  │ • Index all frames   │  │ • Score candidates 1-10     ││
│  │ • Semantic search    │  │ • Pick best per page        ││
│  └──────────────────────┘  └──────────────────────────────┘│
│  ┌──────────────────────┐                                   │
│  │ Reranking (LOCAL)    │                                   │
│  │ Qwen3-VL-Reranker-2B │                                   │
│  │ • Refine selections  │                                   │
│  └──────────────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: PDF Generation                                     │
│  • ReportLab + Pillow                                        │
│  • Ghostscript compression                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    📖 Your Picture Book!
```

---

## Acknowledgments

This project was inspired by research in video understanding and multimodal AI:

- **[TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)** (Ren et al., CVPR 2024) - Time-sensitive multimodal understanding for long videos
- **[VideoAgent](https://wxh1996.github.io/VideoAgent-Website/)** (Wang et al., 2024) - Agentic video understanding with iterative frame selection
- **[FrameExtractor](https://github.com/UpHash-Network/FrameExtractor)** - Intelligent frame extraction techniques

---

## License

MIT License - see [LICENSE](LICENSE) file.
