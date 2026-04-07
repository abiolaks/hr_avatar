# HR Avatar

A real-time conversational AI HR assistant that listens to employee questions, generates intelligent responses using a RAG-powered LLM, synthesizes a voice reply, and renders a lip-synced avatar video — designed to integrate seamlessly with a Learning Management System (LMS).

---

## Architecture

### Standalone mode (microphone)
```
Microphone → VAD → Whisper → HRAgent → XTTS → Wav2Lip → Video
                                  ↕
                            ChromaDB (RAG)
                       HR Policy Documents (PDF/TXT)
```

### LMS-integrated mode (API)
```
LMS Backend ──POST /session/start──► Avatar API
                { user profile }          │
                                    Session Store
                                    (profile + agent)
                                          │
LMS Frontend ──POST /chat──────────►  Agent.run()
              { session_id,              ↕
                message }          ChromaDB (RAG)
                                          │
                                    recommend_courses tool
                                    merges profile + intent
                                          │
                                   Recommendation API
                                   (full payload, no gaps)
                                          │
              ◄── reply + video_url ──────┘
```

| Component | Technology |
|---|---|
| Voice Activity Detection | Silero VAD (browser) / RMS threshold (Python standalone) |
| Speech-to-Text | faster-whisper large-v3 (load-on-demand, unloads after each turn) |
| AI Brain / Agent | Custom HRAgent + ChatOllama (granite4:3b) |
| RAG / Policy Search | ChromaDB + FastEmbedEmbeddings (BAAI/bge-small-en-v1.5, in-process) |
| Text-to-Speech | Coqui XTTS v2 (voice cloning from sample WAV) |
| Lip Sync | Wav2Lip GAN + RetinaFace |
| Web API | FastAPI + Uvicorn |
| LMS Session Layer | In-memory session store + ContextVar profile injection |

---

## Project Structure

```
hr_avatar/
├── main.py                        # Entry point — standalone microphone pipeline
├── config.py                      # Paths, model names, API endpoints, secrets
├── logger.py                      # Logging + performance decorator
├── conftest.py                    # pytest sys.path setup
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container build (CPU/GPU)
├── docker-compose.yml             # Multi-service orchestration
├── blockers_n_solutions.md        # Setup issues and fixes log
│
├── brain/
│   ├── agent.py                   # Custom HRAgent — single-pass tool execution, hallucination guard
│   ├── rag.py                     # ChromaDB RAG manager (PDF, TXT, DOCX — local + Azure)
│   ├── tools.py                   # LangChain tools: retrieve_policy, recommend_courses, generate_assessment
│   ├── session.py                 # In-memory session store (60-min TTL)
│   ├── session_context.py         # ContextVar — injects LMS profile into tools per-request
│   └── state.py                   # Agent state schema
│
├── vad/vad.py                     # Microphone + speech segmentation
├── transcriber/transcriber.py     # faster-whisper wrapper
├── voice/voice.py                 # XTTS voice synthesis
├── face/face.py                   # Wav2Lip lip-sync runner
│
├── web/app.py                     # FastAPI server — LMS integration endpoints
├── mock_services.py               # Mock recommendation + assessment APIs (demo/dev)
│
├── frontend/                      # CEO demo web UI (HTML/CSS/JS)
│   ├── index.html                 # Single-page app
│   ├── style.css                  # Professional styling
│   └── app.js                     # Session, chat, audio, video logic
│
├── eval.py                        # End-to-end evaluation harness (tool routing + quality + latency)
├── assets/                        # Avatar image, video, voice sample, pre-rendered welcome.mp4
├── hr_docs/                       # HR policy documents (PDF, TXT, DOCX)
└── tests/
    ├── test_rag.py                # RAG ingestion and retrieval
    ├── test_tool.py               # Tool unit tests (mocked API calls — no server needed)
    ├── test_session.py            # Session store + API endpoint tests
    └── test_brain.py              # Full agent conversation (requires Ollama)
```

---

## LMS Integration

### How it works

The Avatar is designed so the LMS owns the employee's profile data. The Avatar owns the conversation. Neither duplicates the other's responsibility.

```
┌─────────────────────────────────────────────────────────┐
│        LMS profile (silent — never ask the employee)    │
├─────────────────────────────────────────────────────────┤
│  user_id, name, job_role, department                    │
│  skill_level, known_skills, enrolled_courses            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│        Conversation intent (agent extracts/infers)      │
├─────────────────────────────────────────────────────────┤
│  learning_goal      — agent asks once if not stated     │
│  preferred_difficulty — accept if stated; fall back     │
│                         to skill_level from LMS profile │
│  preferred_duration — INFERRED from time mentions,      │
│                        never asked explicitly           │
│  preferred_category — extracted from topic mentions     │
└─────────────────────────────────────────────────────────┘
```

**Duration inference mapping** (LLM infers without asking):

| Employee says | Maps to |
|---|---|
| "weekends only", "5 hours a week", "not much time" | `Short` |
| "an hour a day", "10 hours a week", "a few hours daily" | `Medium` |
| "full time", "20 hours a week", "intensive" | `Long` |

The `recommend_courses` tool merges both sources before calling the recommendation API, sending a complete payload every time.

---

### API Endpoints

| Method | Path | Caller | Purpose |
|---|---|---|---|
| `POST` | `/session/start` | LMS backend | Start session, pass employee profile |
| `POST` | `/session/welcome` | LMS frontend | Generate personalised spoken welcome (TTS + lip-sync, no LLM) |
| `POST` | `/chat` | LMS frontend | Text conversation turn |
| `POST` | `/chat/audio` | LMS frontend | Audio upload — transcribed then processed |
| `GET` | `/video/{id}` | LMS frontend | Stream the lip-sync video response |
| `DELETE` | `/session/{id}` | LMS backend | End session on logout |
| `GET` | `/health` | LMS / monitoring | Liveness check |
| `POST` | `/admin/ingest` | Admin / CI | Trigger RAG ingestion from local folder and/or Azure Blob Storage |

---

### Session start

The LMS backend calls this once when an employee opens the Avatar widget. The employee profile is stored server-side — the frontend never needs to send it again.

**Request**
```http
POST /session/start
Authorization: Bearer <LMS_SHARED_SECRET>
Content-Type: application/json

{
  "user_id": "emp_9821",
  "name": "Abiola K.",
  "job_role": "Data Analyst",
  "department": "Engineering",
  "skill_level": "Intermediate",
  "known_skills": ["SQL", "Python"],
  "enrolled_courses": ["data-101", "sql-advanced"],
  "context": "dashboard"
}
```

**Response**
```json
{
  "session_id": "sess_abc123def456",
  "message": "Session started for Abiola K."
}
```

The `session_id` is stored by the LMS frontend and included in every subsequent `/chat` request.

---

### Chat turn (text)

```http
POST /chat
Content-Type: application/json

{
  "session_id": "sess_abc123def456",
  "message": "I want to move into machine learning, I only have weekends free"
}
```

**Response**
```json
{
  "session_id": "sess_abc123def456",
  "reply": "Based on your Python background, here are some short weekend-friendly ML courses...",
  "video_url": "/video/output_xyz.mp4"
}
```

The recommendation API receives the full payload automatically:

```json
{
  "user_id": "emp_9821",
  "name": "Abiola K.",
  "job_role": "Data Analyst",
  "department": "Engineering",
  "skill_level": "Intermediate",
  "known_skills": ["SQL", "Python"],
  "enrolled_courses": ["data-101", "sql-advanced"],
  "context": "avatar_chat",
  "learning_goal": "move into machine learning",
  "preferred_difficulty": "Intermediate",
  "preferred_duration": "Short",
  "preferred_category": ""
}
```

---

### Chat turn (audio)

`session_id` must be sent as a **form field** alongside the audio file, not as a query parameter.

```http
POST /chat/audio
Content-Type: multipart/form-data

session_id: sess_abc123def456
audio: <audio file from microphone (WAV or WebM)>
```

The server transcribes the audio then processes it identically to `/chat`.

---

### Security

The shared secret between the LMS and Avatar is set via environment variable:

```bash
export LMS_SHARED_SECRET="your-secret-here"
```

The LMS backend sends it as `Authorization: Bearer <secret>` on `/session/start`. All other endpoints require only a valid `session_id`.

Sessions expire automatically after **60 minutes** of inactivity.

---

## Running the API server

```bash
source hr_venv/bin/activate
ollama serve          # separate terminal

# Start the Avatar API
uvicorn web.app:app --host 0.0.0.0 --port 8000

# Or directly
python web/app.py
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## Running the Frontend Demo

A browser-based demo UI is included in `frontend/`. It provides a full chat interface with avatar video playback — designed for stakeholder demos.

### Start all services

Open four terminals:

**Terminal 1 — Ollama (LLM)**
```bash
ollama serve
```

**Terminal 2 — Avatar API**
```bash
source hr_venv/bin/activate
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

**Terminal 3 — Mock LMS services** (recommendation + assessment APIs)
```bash
source hr_venv/bin/activate
python mock_services.py
```

**Terminal 4 — Frontend**
```bash
cd frontend
python -m http.server 3000
```

Then open **http://localhost:3000** in your browser.

### What the demo shows

| Feature | How to trigger |
|---|---|
| Personalised welcome | Automatic on sign-in — avatar speaks your name, role, and skills |
| HR policy question | Type or ask "What is the annual leave policy?" |
| Course recommendation | Click "Course Recs" quick button or ask naturally |
| Knowledge assessment | Click "Assessment" quick button |
| Voice input (VAD) | VAD activates automatically after the welcome — just speak |
| Mute/unmute VAD | Click the mic button to toggle voice detection on/off |

The avatar loops the silent face video while idle, shows "Thinking…" while processing, and plays the lip-synced response video when ready. Text reply and video arrive simultaneously — neither appears without the other.

#### Voice Activity Detection (VAD)

The frontend uses browser-side VAD powered by the Web Audio API, mirroring the Python `vad/vad.py` behaviour:

- **Auto-starts** after the welcome video finishes — no button press needed
- **Automatically detects** when you start speaking (RMS amplitude threshold)
- **Automatically stops** recording after ~700 ms of silence (matching Python `SILENCE_LIMIT`)
- **Pauses** during backend processing (no accidental re-triggers)
- **Mic button** = mute/unmute toggle, not a push-to-talk button

Mic button states:
| Colour | Meaning |
|---|---|
| Grey | VAD off (muted) |
| Green pulse | Listening — waiting for speech |
| Red pulse | Speech detected — recording |

### Mock services

`mock_services.py` runs a fake recommendation API on port 8001 for demo use. The real recommendation and assessment APIs are to be built by the software engineering team. To point at the real APIs when they are ready:

```bash
export RECOMMENDATION_API_URL=https://your-lms.com/api/recommend
export ASSESSMENT_API_URL=https://your-lms.com/api/generate
```

---

## Platform Support

| Platform | GPU | Notes |
|---|---|---|
| macOS Apple Silicon (M1/M2/M3) | CPU (MPS fallback) | XTTS and Whisper auto-select CPU — MPS unsupported for FFT ops |
| Ubuntu / Windows — NVIDIA GPU | CUDA (auto-detected) | No code changes needed — all components auto-detect CUDA |
| Ubuntu / Windows — CPU only | CPU | Works but slow (~2–4 min per response) |

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
ollama pull granite4:3b
```

### 7. Add HR documents and run

```bash
# Place PDF/TXT/DOCX policy files in hr_docs/
ollama serve          # separate terminal

# Standalone microphone mode (ingests hr_docs/ on startup)
python main.py

# Or LMS API mode
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### 8. (Optional) Ingest documents from Azure Blob Storage

```bash
# Set Azure env vars
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
export AZURE_STORAGE_CONTAINER="hr-documents"

# Trigger ingestion (local + Azure)
curl -X POST http://localhost:8000/admin/ingest \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"local_path": "./hr_docs", "azure_container": "hr-documents"}'
```

Supported document formats: `.pdf`, `.txt`, `.docx`

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

### 6. Download checkpoints and run

Same as macOS steps 4–7 above. No code changes needed — XTTS and Whisper auto-detect CUDA.

---

## Option C — Azure ML Compute Instance (Standard_NC8as_T4_v3)

This is the recommended path for a shared demo or production pilot. The compute instance runs Ubuntu with CUDA pre-installed — no driver setup needed.

### 1. Connect to the instance

**Option A — Azure ML Studio terminal**
In Azure ML Studio → Compute → your instance → click **Terminal**.

**Option B — VS Code (recommended)**
In Azure ML Studio → Compute → your instance → click **VS Code**. This opens a full remote session with integrated port forwarding.

**Option C — SSH**
```bash
ssh azureuser@<your-instance-dns>.instances.azureml.ms
```
Get the SSH command from: Azure ML Studio → Compute → your instance → Connect → SSH instructions.

---

### 2. Verify GPU is available

```bash
nvidia-smi
# Should show: Tesla T4, 16384MiB, CUDA Version 12.x
```

---

### 3. Install system dependencies

```bash
sudo apt update && sudo apt install -y \
    ffmpeg pkg-config portaudio19-dev \
    git curl build-essential libsndfile1
```

---

### 4. Clone the repository

```bash
cd /home/azureuser
git clone https://github.com/abiolaks/hr_avatar.git
cd hr_avatar
```

---

### 5. Create Python environment

Azure ML instances have Python 3.10 available. Check with:
```bash
python3.10 --version
# If not found: sudo apt install python3.10 python3.10-venv
```

```bash
python3.10 -m venv hr_venv
source hr_venv/bin/activate
```

---

### 6. Install PyTorch with CUDA (do this FIRST)

CUDA 12.1 drivers are pre-installed on Azure ML GPU instances. Install the matching PyTorch build before `requirements.txt`:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is visible to PyTorch:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | Tesla T4
```

**Do not proceed if this returns False** — the pipeline will run on CPU and be slow.

---

### 7. Install Python dependencies

```bash
pip install -r requirements.txt

# TTS — bypass numpy pin
pip install TTS==0.22.0 --no-deps
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk "ollama>=0.3.0" "langchain-ollama==0.1.3" \
    setuptools
```

---

### 8. Clone Wav2Lip and download checkpoints

```bash
git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip

# Wav2Lip GAN model (416 MB)
curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
    -o wav2lip_gan.pth

# Face detection model
curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth
```

---

### 9. Install and start Ollama

Azure ML instances do not have Ollama pre-installed.

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama as a background service
ollama serve &

# Wait a few seconds, then pull models
sleep 5
ollama pull granite4:3b
```

Verify Ollama is using the GPU:
```bash
ollama run granite4:3b "hello"
# Watch nvidia-smi in another terminal — GPU utilisation should spike
```

---

### 10. Upload assets

The two asset files must be present — they are not in the repo (binary files).

```bash
mkdir -p assets
```

Copy from your local machine using `scp` or VS Code file explorer:
- `assets/hr_avatar_silent.mp4` — looping silent face video (idle avatar)
- `assets/voice_sample.wav` — voice cloning sample for XTTS (~10–30s of speech)

```bash
# From your LOCAL machine:
scp assets/hr_avatar_silent.mp4  azureuser@<instance-dns>:/home/azureuser/hr_avatar/assets/
scp assets/voice_sample.wav      azureuser@<instance-dns>:/home/azureuser/hr_avatar/assets/
```

---

### 11. Add HR documents

```bash
# Copy your policy PDFs/TXTs into hr_docs/
scp hr_docs/*.pdf azureuser@<instance-dns>:/home/azureuser/hr_avatar/hr_docs/

# Or upload via VS Code file explorer
```

---

### 12. Start all services

Open three terminals (or use `tmux` / `screen` to keep sessions alive):

**Terminal 1 — Ollama (if not already running)**
```bash
ollama serve
```

**Terminal 2 — Avatar API**
```bash
cd /home/azureuser/hr_avatar
source hr_venv/bin/activate
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

**Terminal 3 — Mock LMS services**
```bash
cd /home/azureuser/hr_avatar
source hr_venv/bin/activate
python mock_services.py
```

**Terminal 4 — Frontend**
```bash
cd /home/azureuser/hr_avatar/frontend
python3 -m http.server 3000
```

---

### 13. Access the frontend from your local browser

The compute instance is not publicly accessible on arbitrary ports. Use SSH port forwarding to tunnel the ports to your local machine.

**If using VS Code Remote SSH:**
In the Ports panel (bottom bar → Ports tab), add:
- Port `8000` → forward to `localhost:8000`
- Port `3000` → forward to `localhost:3000`

VS Code does this automatically — just click **Add Port** and enter the number.

**If using plain SSH:**
```bash
ssh -L 8000:localhost:8000 -L 3000:localhost:3000 \
    azureuser@<instance-dns>.instances.azureml.ms
```

Then open **http://localhost:3000** in your browser. Requests to `localhost:8000` are automatically tunnelled to the instance.

---

### 14. Verify GPU is being used

Run this in a separate terminal while the avatar is responding to a message:

```bash
watch -n 1 nvidia-smi
```

You should see:
- GPU utilisation spike during XTTS synthesis and Wav2Lip
- ~7–8GB of 16GB VRAM in use when all models are loaded
- Overall latency ~10–18s vs 2–4 min on CPU

---

### 15. Keep services running when you disconnect

Use `tmux` so sessions survive SSH disconnection:

```bash
# Install tmux (usually pre-installed on Azure ML)
sudo apt install -y tmux

# Start a named session
tmux new -s hravatar

# Inside tmux: start services, then detach with Ctrl+B then D
# Reattach later:
tmux attach -t hravatar
```

---

### Estimated performance on NC8as_T4_v3

| Stage | Time |
|---|---|
| Whisper large-v3 (STT) | ~1–2s |
| Ollama granite4:3b | ~4–8s |
| XTTS v2 | ~4–8s |
| Wav2Lip | ~2–4s |
| **Total end-to-end** | **~10–22s** |

---

### Cost control

The instance charges only when **Running**. Stop it when not in use:

```bash
# From Azure ML Studio → Compute → Stop
# Or via CLI:
az ml compute stop --name <your-instance-name> --workspace-name <ws> --resource-group <rg>
```

At $0.94/hr an 8-hour demo day costs ~$7.50. Leaving it running overnight accidentally costs ~$7. Set an **idle shutdown** policy in Azure ML Studio → Compute → Edit → Idle shutdown.

---

## Option D — Docker (Recommended for Ubuntu + NVIDIA)

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
    -e LMS_SHARED_SECRET=your-secret-here \
    -p 8000:8000 \
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
    -e LMS_SHARED_SECRET=your-secret-here \
    -p 8000:8000 \
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
ENV LMS_SHARED_SECRET=change-me-in-production

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
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
      - LMS_SHARED_SECRET=change-me-in-production
    ports:
      - "8000:8000"
    devices:
      - /dev/snd   # microphone access (standalone mode only)
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
> docker compose exec ollama ollama pull granite4:3b
> docker compose exec ollama ollama pull granite4:3b
> ```

---

## Agent Architecture

The agent (`brain/agent.py`) uses a **single-pass custom implementation** instead of a LangGraph ReAct loop. This was necessary because ReAct's multi-step loop caused runaway tool-call chains with `granite4:3b` (up to 6 chained calls, ~57s latency).

### How a turn works

```
User message
     │
     ▼
llm_with_tools.invoke(messages)          ← single LLM call with tools bound
     │
     ├── response.tool_calls populated?
     │       │
     │       ├── retrieve_policy  → run tool → second LLM call (no tools) to summarise
     │       ├── recommend_courses → run tool → _phrase_tool_result (no second LLM call)
     │       └── generate_assessment → run tool → _phrase_tool_result
     │
     ├── response.content is a bare tool name? ("retrieve_policy")
     │       └── leaked-tool-call fallback → run tool directly
     │
     └── direct LLM answer?
             └── hallucination guard → if response looks like a course list, force recommend_courses
```

### Tool routing rules (enforced in system prompt + code)

| Employee says… | Tool called | Second LLM pass? |
|---|---|---|
| Any HR policy / leave / benefits question | `retrieve_policy` | Yes — summarises multi-paragraph docs to 1–3 sentences |
| Any learning / course / skills request | `recommend_courses` | No — structured list returned directly |
| "I finished [course], test me" | `generate_assessment` | No — question list returned directly |

### Hallucination guard

If the model attempts to answer a course recommendation from memory (response contains "here are some courses", "I recommend", etc.) without calling the tool, the guard detects the pattern and forces a `recommend_courses` call. This prevents the avatar from inventing course names.

---

## Evaluation

`eval.py` runs a fixed set of 15 test cases against the live system and reports:

```bash
source hr_venv/bin/activate
python mock_services.py   # Terminal 1
python eval.py            # Terminal 2 (Ollama must be running)
```

Metrics reported:
- **Tool routing accuracy** — was the correct tool called for each input?
- **Response quality** — does the response contain expected keywords?
- **Hallucination guard fires** — how many times did the safety net intercept?
- **Latency** — per-case, average, and p95

---

## Running Tests

```bash
pytest tests/ -v
```

| Test file | What it tests | Requires |
|---|---|---|
| `test_rag.py` | RAG document ingestion and retrieval | None (FastEmbed is in-process) |
| `test_tool.py` | `recommend_courses` and `generate_assessment` tools | None (API calls mocked) |
| `test_session.py` | Session store, `/session/start`, `/chat` endpoints | None (mocked) |
| `test_brain.py` | HRAgent full conversation | Ollama running |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LMS_SHARED_SECRET` | `dev-secret` | Secret the LMS backend uses to authenticate `/session/start` and `/admin/ingest` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `RECOMMENDATION_API_URL` | `http://localhost:8001/recommend` | LMS recommendation engine |
| `ASSESSMENT_API_URL` | `http://localhost:8001/generate` | LMS assessment engine |
| `AZURE_STORAGE_CONNECTION_STRING` | _(empty)_ | Azure Blob Storage connection string for RAG document ingestion |
| `AZURE_STORAGE_CONTAINER` | `hr-documents` | Azure Blob Storage container name |
| `LOG_LEVEL` | `INFO` | Logging level |

> Set `LMS_SHARED_SECRET` to a strong random value in production. Never commit it.

---

## Example Conversations

### HR Policy questions
- *"What are the standard working hours?"*
- *"How many days of annual leave do I get?"*
- *"How long is paternity leave?"*
- *"What is the code of conduct policy?"*
- *"How much notice do I need to give to resign?"*

### Course recommendations (LMS integrated)
- *"I want to move into machine learning"* — agent asks nothing; uses LMS profile + infers duration from context
- *"I want to get into cloud engineering, I only have weekends"* — infers `Short` duration, sends `preferred_category: cloud`
- *"Something advanced in data science, I can dedicate about 10 hours a week"* — infers `Medium`, difficulty `Advanced`

### Assessments
- *"I just finished the Python Basics course"* — agent extracts the course name as the ID
- *"I finished CS50P, test me"* — maps to `cs50p`, serves real Python questions
- *"I completed the Machine Learning Specialization"* — maps to `machine-learning-specialization`

The course name the employee gives is lowercased and spaces replaced with hyphens to look up the assessment. All 24 courses in the catalog have real questions. Any unrecognised course falls back to a 3-question self-reflection set.

**Assessment course IDs** (say the course name naturally — the system maps it automatically):

| Category | Course | ID |
|---|---|---|
| Python | CS50P: Intro to Programming with Python | `cs50p` |
| Python | Kaggle Python | `kaggle-python` |
| Python | Automate the Boring Stuff with Python | `automate-the-boring-stuff` |
| Python | Python for Everybody Specialization | `python-for-everybody` |
| Python | Python Intermediate: OOP | `python-oop` |
| Machine Learning | Google ML Crash Course | `google-ml-crash-course` |
| Machine Learning | Kaggle: Intro to Machine Learning | `kaggle-intro-to-machine-learning` |
| Machine Learning | Microsoft ML for Beginners | `microsoft-ml-for-beginners` |
| Machine Learning | Machine Learning Specialization (Andrew Ng) | `machine-learning-specialization` |
| Machine Learning | Kaggle: Intermediate Machine Learning | `kaggle-intermediate-machine-learning` |
| Machine Learning | fast.ai: Practical Machine Learning | `fastai-practical-machine-learning` |
| Machine Learning | Stanford CS229 | `stanford-cs229` |
| Deep Learning | Deep Learning Specialization (Andrew Ng) | `deep-learning-specialization` |
| Deep Learning | MIT 6.S191: Intro to Deep Learning | `mit-intro-deep-learning` |
| Deep Learning | fast.ai: Deep Learning from the Foundations | `fastai-deep-learning-foundations` |
| AI Agents | Introduction to AI Agents | `introduction-to-ai-agents` |
| AI Agents | AI Agents in LangGraph | `ai-agents-in-langgraph` |
| AI Agents | LangChain for LLM Application Development | `langchain-for-llm-application-development` |
| AI Agents | Functions, Tools and Agents with LangChain | `functions-tools-and-agents-with-langchain` |
| AI Agents | Building Agentic RAG with LlamaIndex | `building-agentic-rag-with-llamaindex` |
| Data Science | Kaggle: Pandas | `kaggle-pandas` |
| Data Science | Kaggle: Data Visualisation | `kaggle-data-visualisation` |
| Data Science | Data Analysis with Python (freeCodeCamp) | `data-analysis-with-python` |
| Data Science | Kaggle: Feature Engineering | `kaggle-feature-engineering` |

---

## Known Limitations

- **CPU response time**: ~2–4 minutes end-to-end (LLM ~30s + XTTS ~90s + Wav2Lip ~60s). Use an NVIDIA GPU for ~10–20s responses.
- **GPU requirement**: Apple MPS is unsupported for XTTS (FFT ops) and Wav2Lip. NVIDIA CUDA only for GPU acceleration.
- **Session store**: In-memory — sessions lost on server restart. Use Redis for production.
- **Video storage**: Lip-sync videos stored in `/tmp` — lost on restart. Use S3 or a persistent volume for production.
- **Recommendation/Assessment APIs**: `mock_services.py` is used in dev. The real LMS services must be built and pointed to via environment variables.
- **Microphone in Docker**: Requires `/dev/snd` device passthrough.
- **LLM tool reliability**: `granite4:3b` performs best with clear, direct questions. Very short or ambiguous inputs may not reliably trigger the correct tool.
