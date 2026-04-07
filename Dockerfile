# =============================================================================
# HR Avatar — Dockerfile
#
# Supports GPU (CUDA 12.1) out of the box.
# For CPU-only machines, change the FROM line to:
#   FROM python:3.10-slim
# and remove the nvidia runtime section in docker-compose.yml.
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip

WORKDIR /app

# ── Python dependencies (layered for cache efficiency) ───────────────────────
# Step 1: main requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: TTS — must be installed after, with --no-deps to bypass numpy pin
RUN pip install --no-cache-dir TTS==0.22.0 --no-deps

# Step 3: Clone Wav2Lip (not a pip package)
RUN git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Runtime directories (created by config.py too, but ensure they exist in image)
RUN mkdir -p assets hr_docs chroma_db logs

EXPOSE 8000

# Default: run the FastAPI web server
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
