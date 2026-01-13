# Vid2BedtimeStory

Convert children's TV episodes into **print-ready picture book PDFs**.

Takes a video file with subtitles → outputs a ~30-page illustrated storybook for ages 5-8.

---

## What You Need

| Requirement | Details |
|-------------|---------|
| **Python 3.10+** | `python3 --version` to check |
| **FFmpeg** | Video processing (`brew install ffmpeg` on Mac, `apt install ffmpeg` on Linux) |
| **Ghostscript** | PDF compression (`brew install ghostscript` on Mac, `apt install ghostscript` on Linux) |
| **OpenRouter API Key** | Create account at [openrouter.ai](https://openrouter.ai), add credits, generate key |
| **A franchise JSON file** | Character database for your show (see below) |

### Platform Support

| Platform | Status |
|----------|--------|
| **Mac (Apple Silicon, 32GB+ RAM)** | ✅ Works out of the box |
| **Mac (Intel) / PC / Linux** | ⚠️ Requires code modification |

**Why Mac Apple Silicon?** The video analysis stage runs a 32B parameter vision model locally to process your video file. Cloud APIs can't accept 500MB+ video uploads, so local inference is required.

The codebase uses Apple's [MLX framework](https://github.com/ml-explore/mlx) for this. **Non-Mac users** would need to replace the MLX worker scripts (`mlx_worker.py`, `mlx_image_worker.py`, etc.) with a PyTorch/Transformers equivalent.

### About API Costs

This tool uses cloud AI models via OpenRouter:
- **Claude Sonnet** for story writing  
- **Qwen VL 235B** for frame selection

You pay per use based on OpenRouter pricing. A typical 11-minute episode costs roughly $1-3 in API calls.

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/vid2bedtimestory/vid2bedtimestory.git
cd vid2bedtimestory
./setup_mac.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Add your OpenRouter API key
echo "sk-or-v1-your-key-here" > openrouterapikey.md

# 4. Run it
python -m vid2bedtimestory build videos/episode.mkv --franchise hot_wheels_lets_race
```

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
| `--subtitle-context N` | Subtitle lines for gap detection (default: 50, see note below) |
| `--rebuild-from STAGE` | Force rebuild from: `subtitles`, `analysis`, `story`, `pagination`, `screenshots`, `pdf` |
| `--fresh` | Delete all artifacts and start over |
| `--dry-run` | Show plan without executing |
| `--keep-candidates` | Keep candidate frames for debugging |
| `--artifacts-dir PATH` | Custom artifacts directory |

### Note on Video Length

This tool was tested primarily on **10-11 minute** episodes. The `--subtitle-context` setting controls how much dialogue the AI sees when detecting gaps in video coverage.

For **longer episodes** (15+ minutes), you may need to increase this value:

```bash
# For a 20-minute episode
python -m vid2bedtimestory build long_episode.mkv --franchise your_show --subtitle-context 100

# For a 30-minute episode  
python -m vid2bedtimestory build long_episode.mkv --franchise your_show --subtitle-context 150
```

A rough guideline: ~5 lines per minute of video (e.g., 20 minutes → 100 lines).

---

## Troubleshooting

### "No subtitle stream found"
Your video doesn't have embedded subtitles. Use `--srt` with an external subtitle file.

### "Franchise not found"
Run `--franchise list` to see available options, or create your own JSON file.

### "openrouterapikey.md not found"
Create the file with your API key:
```bash
echo "sk-or-v1-your-key-here" > openrouterapikey.md
```

### Pipeline seems stuck
The video analysis stage can take 3-5 minutes. Story and pagination are faster (~30 seconds each). Frame selection is the longest (8-12 minutes).

### Out of memory
This tool requires 32GB RAM. Close other applications if you're running low.

---

## How It Works (Overview)

1. **Subtitles** - Extract dialogue from video
2. **Analysis** - Local AI scans video frames for scenes and characters
3. **Story** - Cloud AI writes a children's story based on analysis
4. **Pagination** - Split story into ~30 pages
5. **Screenshots** - Cloud AI picks the best frame for each page
6. **PDF** - Render the final book

Total time: ~15-20 minutes per episode.

---

## Acknowledgments

This project was inspired by research in video understanding and multimodal AI:

- **[TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)** (Ren et al., CVPR 2024) - Time-sensitive multimodal understanding for long videos. Influenced our approach to timestamp-aware video analysis.

- **[VideoAgent](https://wxh1996.github.io/VideoAgent-Website/)** (Wang et al., 2024) - Agentic video understanding with iterative frame selection. Inspired our multi-phase analysis pipeline.

- **[FrameExtractor](https://github.com/UpHash-Network/FrameExtractor)** - Intelligent frame extraction techniques. Influenced our VLM-based frame scoring approach.

---

## License

MIT License - see [LICENSE](LICENSE) file.
