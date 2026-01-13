#!/bin/bash
# ============================================
# Vid2BedtimeStory - Mac Setup Script
# ============================================
# This script sets up an isolated Python virtual environment
# with all dependencies for running vid2bedtimestory on Apple Silicon
# ============================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Vid2BedtimeStory - Mac Setup${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Working directory: ${SCRIPT_DIR}${NC}"
echo ""

# ============================================
# Step 1: Check for Homebrew
# ============================================
echo -e "${BLUE}[1/5] Checking for Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo -e "${GREEN}✓ Homebrew found${NC}"
fi

# ============================================
# Step 2: Install FFmpeg
# ============================================
echo ""
echo -e "${BLUE}[2/5] Checking for FFmpeg...${NC}"
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}FFmpeg not found. Installing via Homebrew...${NC}"
    brew install ffmpeg
else
    echo -e "${GREEN}✓ FFmpeg found: $(which ffmpeg)${NC}"
    ffmpeg -version | head -1
fi

# ============================================
# Step 3: Check Python version
# ============================================
echo ""
echo -e "${BLUE}[3/5] Checking Python version...${NC}"

# Prefer python3.11 or python3.12 for MLX compatibility
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [[ $major -eq 3 ]] && [[ $minor -ge 10 ]]; then
            PYTHON_CMD=$cmd
            echo -e "${GREEN}✓ Found $cmd (version $version)${NC}"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo -e "${RED}ERROR: Python 3.10+ required but not found${NC}"
    echo -e "${YELLOW}Install Python via Homebrew:${NC}"
    echo "  brew install python@3.12"
    exit 1
fi

# ============================================
# Step 4: Create Virtual Environment
# ============================================
echo ""
echo -e "${BLUE}[4/5] Creating virtual environment...${NC}"

VENV_DIR="$SCRIPT_DIR/.venv"

if [[ -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}Virtual environment already exists at .venv${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    else
        echo -e "${GREEN}✓ Using existing virtual environment${NC}"
    fi
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at .venv${NC}"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ============================================
# Step 5: Install Dependencies
# ============================================
echo ""
echo -e "${BLUE}[5/5] Installing dependencies...${NC}"

# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
echo -e "${YELLOW}Installing core dependencies...${NC}"
pip install -r requirements.txt

# Verify installations
echo ""
echo -e "${BLUE}Verifying installations...${NC}"
python -c "import reportlab; print(f'✓ reportlab {reportlab.Version}')"
python -c "import PIL; print(f'✓ pillow {PIL.__version__}')"
python -c "import srt; print(f'✓ srt {srt.__version__}')"
python -c "import pydantic; print(f'✓ pydantic {pydantic.VERSION}')"
python -c "import typer; print(f'✓ typer {typer.__version__}')"
python -c "import rich; print(f'✓ rich {rich.__version__}')"
python -c "import requests; print(f'✓ requests {requests.__version__}')"

# Check MLX (may fail on non-Apple Silicon)
echo ""
echo -e "${YELLOW}Checking MLX installation (Apple Silicon only)...${NC}"
if python -c "import mlx" 2>/dev/null; then
    python -c "import mlx; print(f'✓ mlx installed')"
    if python -c "import mlx_vlm" 2>/dev/null; then
        python -c "import mlx_vlm; print(f'✓ mlx-vlm installed')"
    else
        echo -e "${YELLOW}⚠ mlx-vlm not installed (video analysis won't work)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ MLX not available (requires Apple Silicon Mac)${NC}"
fi

# Check PyTorch and Transformers (required for embedding/reranking)
echo ""
echo -e "${YELLOW}Checking PyTorch installation (for embedding models)...${NC}"
if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print(f'✓ torch {torch.__version__}')"
    # Check MPS availability
    if python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        echo -e "${GREEN}✓ MPS (Metal Performance Shaders) available${NC}"
    else
        echo -e "${YELLOW}⚠ MPS not available (will use CPU for embeddings)${NC}"
    fi
else
    echo -e "${RED}✗ torch not installed (embedding models won't work)${NC}"
fi

if python -c "import transformers" 2>/dev/null; then
    python -c "import transformers; print(f'✓ transformers {transformers.__version__}')"
else
    echo -e "${RED}✗ transformers not installed (embedding models won't work)${NC}"
fi

# ============================================
# Complete!
# ============================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${BLUE}To activate the virtual environment:${NC}"
echo "  cd $SCRIPT_DIR"
echo "  source .venv/bin/activate"
echo ""
echo -e "${BLUE}To run vid2bedtimestory:${NC}"
echo "  python -m vid2bedtimestory test                    # Test CLI"
echo "  python -m vid2bedtimestory build --help            # Show build options"
echo "  python -m vid2bedtimestory build <video> --franchise <id>  # Build a book"
echo ""
echo -e "${BLUE}To deactivate the virtual environment:${NC}"
echo "  deactivate"
echo ""
echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}  IMPORTANT: First Run Info${NC}"
echo -e "${YELLOW}============================================${NC}"
echo ""
echo -e "On your first run, AI models will auto-download (~26GB total):"
echo "  • Qwen3-VL-32B-Instruct-8bit (~18GB) - Video analysis"
echo "  • Qwen3-VL-Embedding-2B (~4GB) - Frame search"
echo "  • Qwen3-VL-Reranker-2B (~4GB) - Frame selection"
echo ""
echo -e "Models are cached in ~/.cache/huggingface/hub/"
echo -e "First run may take 10-15 extra minutes for downloads."
echo ""
echo -e "${YELLOW}Don't forget to add your OpenRouter API key:${NC}"
echo "  echo \"sk-or-v1-your-key-here\" > openrouterapikey.md"
echo ""

