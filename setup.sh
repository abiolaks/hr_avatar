#!/usr/bin/env bash
# =============================================================================
# HR Avatar — Local Setup Script
#
# Run once on a fresh clone to set up the full local dev environment.
# Requires: Python 3.10 or 3.11, git, ffmpeg, Ollama, CUDA toolkit (for GPU)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── 1. Python version check ──────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_BIN=""
for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=${version%.*}; minor=${version#*.}
        if [[ "$major" -eq 3 && "$minor" -ge 10 ]]; then
            PYTHON_BIN="$candidate"
            info "Using $PYTHON_BIN ($version)"
            break
        fi
    fi
done
[[ -z "$PYTHON_BIN" ]] && error "Python 3.10+ required. Install it and re-run."

# ── 2. Create virtual environment ────────────────────────────────────────────
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 3. Install Python dependencies ───────────────────────────────────────────
info "Step 1/3: Installing requirements.txt..."
pip install -r requirements.txt

info "Step 2/3: Installing TTS==0.22.0 (--no-deps to bypass numpy pin)..."
pip install TTS==0.22.0 --no-deps

info "Step 3/3: Cloning Wav2Lip..."
if [[ ! -d "face/wav2lip" ]]; then
    git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip
    info "Wav2Lip cloned."
else
    info "face/wav2lip already exists — skipping clone."
fi

# ── 4. Download wav2lip_gan.pth checkpoint ───────────────────────────────────
CHECKPOINT="wav2lip_gan.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
    info "Downloading wav2lip_gan.pth from HuggingFace (~400 MB)..."
    curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
         -o "$CHECKPOINT" \
         --progress-bar
    info "Checkpoint saved to $CHECKPOINT"
else
    info "$CHECKPOINT already exists — skipping download."
fi

# ── 5. Environment file ───────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    warn ".env created from .env.example — edit it before running in production."
else
    info ".env already exists."
fi

# ── 6. Ollama check ───────────────────────────────────────────────────────────
info "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    warn "Ollama not found. Install it from https://ollama.com then run:"
    warn "  ollama pull granite4:3b"
else
    info "Ollama found. Pulling granite4:3b model (skips if already present)..."
    ollama pull granite4:3b
fi

# ── 7. Assets check ───────────────────────────────────────────────────────────
info "Checking required assets..."
MISSING=0
for f in assets/hr_avatar.jpg assets/hr_voice_sample.wav assets/hr_avatar_silent.mp4; do
    if [[ ! -f "$f" ]]; then
        warn "Missing asset: $f"
        MISSING=1
    fi
done
[[ "$MISSING" -eq 1 ]] && warn "Add the missing assets before running the server."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Activate the venv:   source .venv/bin/activate"
echo "  Start mock services: python mock_services.py &"
echo "  Start the server:    uvicorn web.app:app --host 0.0.0.0 --port 8000"
echo "  Standalone mode:     python main.py"
echo ""
