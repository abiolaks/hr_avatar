# HR Avatar

A real-time conversational AI HR assistant that listens to employee questions, generates intelligent responses using a RAG-powered LLM, synthesizes a voice reply, and renders a lip-synced avatar video — all running locally.

---

## Architecture

```
Microphone → VAD → Whisper → LangGraph Agent → XTTS → Wav2Lip → Video
                                    ↕
                              ChromaDB (RAG)
                         HR Policy Documents (PDF/TXT)
```

| Component | Technology |
|---|---|
| Voice Activity Detection | Silero VAD |
| Speech-to-Text | faster-whisper (base model) |
| AI Brain / Agent | LangGraph + ChatOllama (qwen3:4b) |
| RAG / Policy Search | ChromaDB + OllamaEmbeddings (nomic-embed-text) |
| Text-to-Speech | Coqui XTTS v2 (voice cloning) |
| Lip Sync | Wav2Lip GAN |

---

## Project Structure

```
hr_avatar/
├── main.py                        # Entry point — orchestrates full pipeline
├── config.py                      # Paths, model names, API endpoints
├── logger.py                      # Logging + performance decorator
├── conftest.py                    # pytest sys.path setup
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container build (CPU/GPU)
├── docker-compose.yml             # Multi-service orchestration
├── blockers_n_solutions.md        # Setup issues and fixes log
│
├── brain/
│   ├── agent.py                   # LangGraph ReAct agent
│   ├── rag.py                     # ChromaDB RAG manager
│   ├── tools.py                   # LangChain tools
│   └── state.py                   # Agent state
│
├── vad/vad.py                     # Microphone + speech segmentation
├── transcriber/transcriber.py     # faster-whisper wrapper
├── voice/voice.py                 # XTTS voice synthesis
├── face/face.py                   # Wav2Lip lip-sync runner
│
├── assets/                        # Avatar image, video, voice sample
├── hr_docs/                       # HR policy documents (PDF or TXT)
└── tests/
    ├── test_rag.py
    ├── test_tool.py
    └── test_brain.py
```

---

## Platform Support

| Platform | GPU | Tested |
|---|---|---|
| macOS Apple Silicon (M1/M2/M3) | MPS (CPU fallback for XTTS) | Yes |
| Ubuntu 20.04+ | NVIDIA CUDA | Yes (via Docker) |
| Ubuntu 20.04+ | CPU only | Yes (slow) |

---

## Option A — Local Setup (macOS Apple Silicon)

### 1. Prerequisites

- Python 3.10
- [Homebrew](https://brew.sh)
- [Ollama](https://ollama.com) installed

```bash
brew install pkg-config ffmpeg
```

### 2. Clone and create virtual environment

```bash
git clone https://github.com/abiolaks/hr_avatar.git
cd hr_avatar
python3.10 -m venv hr_venv
source hr_venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# TTS needs special install (numpy version conflict)
pip install TTS==0.22.0 --no-deps
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk "ollama>=0.3.0" "langchain-ollama==0.1.3" \
    setuptools
```

### 4. Clone Wav2Lip

```bash
git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip
```

### 5. Download model checkpoints

```bash
# Wav2Lip GAN (416MB)
curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
    -o wav2lip_gan.pth

# Face detection model
curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth
```

### 6. Pull Ollama models

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 7. Add HR documents and run

```bash
# Place PDF/TXT policy files in hr_docs/
ollama serve          # separate terminal
python main.py
```

---

## Option B — Local Setup (Ubuntu + NVIDIA GPU)

### 1. Prerequisites

```bash
sudo apt update && sudo apt install -y python3.10 python3.10-venv \
    ffmpeg pkg-config portaudio19-dev git curl build-essential

# Install CUDA 12.1+ from https://developer.nvidia.com/cuda-downloads
# Verify:
nvidia-smi
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Clone and create virtual environment

```bash
git clone https://github.com/abiolaks/hr_avatar.git
cd hr_avatar
python3.10 -m venv hr_venv
source hr_venv/bin/activate
```

### 4. Install PyTorch with CUDA support

> Replace the default torch in requirements.txt with the CUDA build:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt

pip install TTS==0.22.0 --no-deps
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk "ollama>=0.3.0" "langchain-ollama==0.1.3" \
    setuptools
```

### 6. Enable GPU in config

Update `voice/voice.py` — NVIDIA CUDA supports all XTTS operations:

```python
# Change this line in voice/voice.py
self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 7. Download checkpoints and run

Same as macOS steps 4–7 above.

---

## Option C — Docker (Recommended for Ubuntu + NVIDIA)

Docker provides a consistent environment across machines without manual dependency management.

### Prerequisites

- Docker Engine 24+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)
- Ollama running on the host

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build the image

```bash
docker build -t hr-avatar .
```

### Run with NVIDIA GPU

```bash
docker run --gpus all \
    --device /dev/snd \
    -e OLLAMA_HOST=http://host.docker.internal:11434 \
    -v $(pwd)/hr_docs:/app/hr_docs \
    -v $(pwd)/assets:/app/assets \
    -v $(pwd)/wav2lip_gan.pth:/app/wav2lip_gan.pth \
    hr-avatar
```

### Run CPU-only

```bash
docker run \
    --device /dev/snd \
    -e OLLAMA_HOST=http://host.docker.internal:11434 \
    -v $(pwd)/hr_docs:/app/hr_docs \
    -v $(pwd)/assets:/app/assets \
    -v $(pwd)/wav2lip_gan.pth:/app/wav2lip_gan.pth \
    hr-avatar
```

### docker-compose (recommended)

```bash
docker compose up
```

> See `docker-compose.yml` for the full multi-service config including Ollama.

---

## Dockerfile

```dockerfile
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg pkg-config portaudio19-dev git curl \
    build-essential libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir TTS==0.22.0 --no-deps && \
    pip install --no-cache-dir coqpit trainer "transformers==4.44.2" einops \
        encodec unidecode inflect num2words pysbd anyascii spacy batch-face \
        pypdf "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
        hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
        umap-learn cython flask g2pkk "ollama>=0.3.0" "langchain-ollama==0.1.3" \
        setuptools

# Clone Wav2Lip
RUN git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip

# Download face detection model
RUN curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth

ENV OLLAMA_HOST=http://host.docker.internal:11434

CMD ["python", "main.py"]
```

---

## docker-compose.yml

```yaml
version: "3.9"

services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  hr-avatar:
    build: .
    depends_on:
      - ollama
    environment:
      - OLLAMA_HOST=http://ollama:11434
    devices:
      - /dev/snd   # microphone access
    volumes:
      - ./hr_docs:/app/hr_docs
      - ./assets:/app/assets
      - ./wav2lip_gan.pth:/app/wav2lip_gan.pth
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

> **Note:** After first `docker compose up`, pull Ollama models inside the container:
> ```bash
> docker compose exec ollama ollama pull qwen3:4b
> docker compose exec ollama ollama pull nomic-embed-text
> ```

---

## Running Tests

```bash
pytest tests/ -v
```

| Test file | What it tests |
|---|---|
| `test_rag.py` | RAG document ingestion and retrieval (requires Ollama) |
| `test_tool.py` | `recommend_courses` and `generate_assessment` tools (mocked) |
| `test_brain.py` | HRAgent full conversation (requires Ollama) |

---

## Example Questions

Based on the Aurora Analytics HR Policy Manual:

- *"What are the standard working hours?"*
- *"How many days of annual leave do I get?"*
- *"How long is paternity leave?"*
- *"What is the code of conduct policy?"*
- *"How much notice do I need to give to resign?"*
- *"Does the company sponsor training and certifications?"*

---

## Known Limitations

- XTTS on CPU takes ~30 seconds per response (use GPU for faster synthesis)
- Wav2Lip adds ~10 seconds per response
- Microphone access inside Docker requires `/dev/snd` device passthrough
- `qwen3:4b` tool reliability improves with direct, specific questions
