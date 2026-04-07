# Blockers & Solutions

A running log of issues encountered during setup and development, with their fixes.

---

## 1. `wav2lip` — Repository Not Found

**Error:**
```
remote: Repository not found.
fatal: repository 'https://github.com/numz/wav2lip.git/' not found
```

**Cause:** The `numz/wav2lip` GitHub repository was deleted/made private.

**Fix:** `wav2lip` is not a pip-installable package (no `setup.py` / `pyproject.toml`). Clone it manually:
```bash
git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip
```
Remove it from `requirements.txt` entirely. `face/__init__.py` adds it to `sys.path` automatically.

---

## 2. `langgraph` — Conflicting `langchain-core` Versions

**Error:**
```
langchain 0.2.0 depends on langchain-core>=0.2.0
langgraph 0.0.20 depends on langchain-core<0.2.0
```

**Cause:** `langgraph 0.0.20` was designed for an older `langchain-core` incompatible with `langchain 0.2.0`.

**Fix:** Upgrade `langgraph` to `0.1.19`, which targets `langchain-core>=0.2.0`.

---

## 3. `av` — Cython Compile Error

**Error:**
```
Cannot assign type '...' to '...'. Exception values are incompatible.
Suggest adding 'noexcept' to the type of 'log_context_name'.
```

**Cause:** `faster-whisper 0.9.0` pulls in `av==10.0.0`, which fails to compile against newer Cython.

**Fix:** Upgrade `faster-whisper` to `1.0.3`, which depends on `av>=12.0.0` where the Cython issue is resolved.

---

## 4. `pkg-config` Missing — PyAV Build Failure

**Error:**
```
pkg-config is required for building PyAV
```

**Cause:** `pkg-config` and `ffmpeg` were not installed on the system.

**Fix:**
```bash
brew install pkg-config ffmpeg
```

---

## 5. `TTS` — Exact `numpy` Pin Conflicts with `chromadb`

**Error:**
```
tts 0.22.0 depends on numpy==1.22.0; python_version <= "3.10"
chromadb 0.4.22 depends on numpy>=1.22.5
```

**Cause:** `TTS 0.22.0` hard-pins `numpy==1.22.0` exactly on Python ≤ 3.10. No single numpy version satisfies both.

**Fix:** Remove `TTS` from `requirements.txt`. Install separately bypassing its numpy pin:
```bash
pip install TTS==0.22.0 --no-deps
```
TTS works fine at runtime with numpy ≥ 1.22.5.

---

## 6. `torch` — No Matching Distribution + `facenet-pytorch` Conflict

**Error:**
```
facenet-pytorch 2.6.0 depends on torch>=2.2.0,<2.3.0
No matching distributions available for torch==2.0.1 (environment: macOS arm64)
```

**Cause:** `torch 2.0.1` has no wheel for macOS Apple Silicon, and `facenet-pytorch 2.6.0` requires `torch>=2.2.0`.

**Fix:** Upgrade the torch stack:
```
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
```

---

## 7. `logger.py` — Missing `os` Import and `Path` Division Error

**Error:**
```
NameError: name 'os' is not defined
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

**Cause:** `logger.py` used `os.path.join` without importing `os`, and used `LOGS_DIR / "file.log"` (Path division syntax) on a plain string.

**Fix:**
- Add `import os` at the top of `logger.py`
- Change `LOGS_DIR / "hr_avatar.log"` → `os.path.join(LOGS_DIR, "hr_avatar.log")`

---

## 8. `ModuleNotFoundError` — Package Imports Inside `__init__.py`

**Error:**
```
ImportError: cannot import name 'VADetector' from 'vad'
```

**Cause:** `vad/__init__.py` used `from vad.vad import VADetector` — absolute import inside a package causes a circular naming conflict.

**Fix:** Use relative imports in all `__init__.py` files:
```python
from .vad import VADetector       # vad/__init__.py
from .voice import VoiceSynthesizer  # voice/__init__.py
from .transcriber import Transcriber  # transcriber/__init__.py
from .face import LipSyncGenerator   # face/__init__.py
```

---

## 9. `pkg_resources` Not Found — `librosa` Import Failure

**Error:**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause:** `librosa 0.10.0` uses `pkg_resources.resource_filename` which is broken in newer environments even when `setuptools` is installed.

**Fix:** Upgrade librosa:
```bash
pip install "librosa==0.10.2"
```

---

## 10. TTS Missing Runtime Dependencies (`--no-deps` side effect)

**Error (repeated):**
```
ModuleNotFoundError: No module named 'pysbd'
ModuleNotFoundError: No module named 'trainer'
ModuleNotFoundError: No module named 'spacy'
...
```

**Cause:** Installing `TTS==0.22.0 --no-deps` skips all its dependencies. They must be installed manually.

**Fix:** Install all TTS runtime deps explicitly:
```bash
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk
```

---

## 11. `transformers` — `BeamSearchScorer` Removed in 4.47+

**Error:**
```
ImportError: cannot import name 'BeamSearchScorer' from 'transformers'
```

**Cause:** `transformers 4.57.x` removed `BeamSearchScorer` from the public API. TTS 0.22.0 uses it internally via XTTS.

**Fix:** Pin transformers to the last version that still exposes it:
```bash
pip install "transformers==4.44.2"
```

---

## 12. `ChatOllama` — `bind_tools` Not Implemented

**Error:**
```
AttributeError: 'Ollama' object has no attribute 'bind_tools'
NotImplementedError (from ChatOllama in langchain-community)
```

**Cause:** `langchain_community.llms.Ollama` is a legacy class with no tool support. `ChatOllama` in `langchain-community==0.2.0` also doesn't implement `bind_tools`.

**Fix:** Use the dedicated `langchain-ollama` package:
```bash
pip install "ollama>=0.3.0" "langchain-ollama==0.1.3"
```
Update import in `brain/agent.py`:
```python
from langchain_ollama import ChatOllama
```

---

## 13. MPS Device — `aten::_fft_r2c` Not Implemented

**Error:**
```
NotImplementedError: The operator 'aten::_fft_r2c' is not currently implemented for the MPS device.
```

**Cause:** XTTS uses FFT operations during speaker embedding that are not supported on Apple MPS (Metal GPU) in PyTorch 2.2.x.

**Fix:** Force XTTS to run on CPU in `voice/voice.py`:
```python
self.device = "cpu"  # MPS does not support aten::_fft_r2c needed by XTTS
```

---

## 14. Wav2Lip — `No module named 'wav2lip'`

**Error:**
```
/usr/bin/python: Error while finding module specification for 'wav2lip.inference'
ModuleNotFoundError: No module named 'wav2lip'
```

**Cause:** `face/face.py` called `python -m wav2lip.inference` but Wav2Lip is not an installed package.

**Fix:** Call `inference.py` as a direct script path:
```python
inference_script = os.path.join(WAV2LIP_DIR, "inference.py")
cmd = ["python", inference_script, ...]
```

---

## 15. Wav2Lip — `checkpoints/mobilenet.pth` Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/mobilenet.pth'
```

**Cause:** Two issues combined:
1. `mobilenet.pth` (face detection model) was never downloaded
2. The subprocess ran from the project root so the relative path `checkpoints/` resolved incorrectly

**Fix:**
1. Download the model:
```bash
curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth
```
2. Run subprocess from the wav2lip directory and use absolute path for checkpoint:
```python
subprocess.run(cmd, check=True, cwd=WAV2LIP_DIR)
self.checkpoint = os.path.abspath(checkpoint_path)
```

---

## 16. RAG Ingesting 0 Documents

**Symptom:** Logs showed `Loaded 0 documents` even though `hr_docs/` had a PDF file.

**Cause:** `brain/rag.py` only loaded `**/*.txt` files. The HR policy document is a `.pdf`.

**Fix:** Add PDF loading support:
```python
from langchain_community.document_loaders import PyPDFLoader
pdf_loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents += pdf_loader.load()
```
Also install: `pip install pypdf`

---

## 17. Wav2Lip Checkpoint — Google Drive / SharePoint Links Dead

**Error:**
```
gdown.exceptions.FileURLRetrievalError: Cannot retrieve the public link of the file.
404 FILE NOT FOUND (SharePoint link)
```

**Cause:** Both the original Google Drive ID in `face/face.py` and the SharePoint link from the Wav2Lip README were dead/expired.

**Fix:** Download from HuggingFace instead:
```bash
curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
    -o wav2lip_gan.pth
```

---

## 18. `recommend_courses` Tool — Asking for Data the LMS Already Has

**Problem:**
The `recommend_courses` tool was asking the employee for `current_role`, `desired_role`, `skills_to_develop`, and `time_commitment`. The LMS already holds the employee's profile (job role, department, skill level, enrolled courses, known skills). Asking the user to repeat this was redundant and poor UX.

**Root Cause:**
The tool was designed as a standalone function with no access to session context. It treated all parameters as things to gather from the conversation.

**Fix:**
Split payload fields into two categories:

1. **LMS profile fields** — injected silently via `brain/session_context.py` using Python's `ContextVar`. The LMS passes these once at `/session/start`; tools read them without asking.
2. **Conversation intent fields** — only what the agent actually extracts from what the employee says.

```
LMS profile (silent):     user_id, name, job_role, department,
                          skill_level, known_skills, enrolled_courses

Conversation intent:      learning_goal (ask once),
                          preferred_difficulty (accept if stated),
                          preferred_duration (infer — never ask),
                          preferred_category (extract if mentioned)
```

Key files changed: `brain/session_context.py` (new), `brain/session.py` (new), `brain/tools.py`, `brain/agent.py`, `web/app.py`.

---

## 19. `preferred_duration` — Agent Was Not Inferring It

**Problem:**
`preferred_duration` (`Short`, `Medium`, `Long`) was being left blank unless the employee explicitly used those words. The recommendation API always received an empty string.

**Root Cause:**
The tool docstring didn't tell the LLM how to map natural language time mentions to the three valid values.

**Fix:**
Add an explicit inference mapping to the tool docstring so the LLM fills the parameter correctly before invoking the tool:

```python
"""
preferred_duration (OPTIONAL — INFER from the conversation, do not ask):
  Map time mentions to one of three values:
    "Short"  — under 3 hours/week ("weekends only", "5 hours a week",
               "not much time")
    "Medium" — 3–10 hours/week ("a few hours daily", "about an hour a day",
               "10 hours a week")
    "Long"   — 10+ hours/week ("full time", "20 hours a week", "intensive")
  If the employee made no time mention at all, leave as None.
"""
```

The LLM reads tool docstrings as part of its tool schema. Inference rules in the docstring are enough — no additional Python logic is needed.

Default when no time is mentioned: `"Short"` (safe fallback in the tool body).

---

## 20. `preferred_difficulty` — Not Falling Back to LMS Skill Level

**Problem:**
When the employee didn't state a difficulty preference, `preferred_difficulty` was sent as an empty string to the recommendation API instead of using the employee's known skill level.

**Root Cause:**
The tool was not reading the session profile to find a sensible default.

**Fix:**
Fall back to the `skill_level` field from the LMS profile when `preferred_difficulty` is not provided by the conversation:

```python
"preferred_difficulty": preferred_difficulty or profile.get("skill_level", "Beginner"),
```

This means:
- Employee says "something advanced" → `"Advanced"` (explicit, wins)
- Employee says nothing about difficulty → `"Intermediate"` (from LMS profile)
- No profile available at all → `"Beginner"` (safe fallback)

---

## 21. `requests.exceptions.RequestException` Not Catching General Exceptions in Tests

**Error:**
```
FAILED tests/test_tool.py::TestRecommendCourses::test_api_error_returns_friendly_message
Exception: Connection refused
```

**Cause:**
Tools caught only `requests.exceptions.RequestException`. The mock raised a plain `Exception`, which is not a subclass of `RequestException` and therefore propagated uncaught — crashing the test instead of returning a friendly error string.

**Fix:**
Broaden the catch clause to `Exception`. LangChain tools should never raise — they must always return a string so the agent can decide what to do:

```python
except Exception as e:
    logger.error(f"Recommendation API error: {e}")
    return f"Sorry, I couldn't fetch recommendations due to a service error: {str(e)}"
```

This also protects against `JSONDecodeError`, `TimeoutError`, and other non-request failures.

---

## 22. Pydantic v2 Deprecation — `.dict()` Replaced by `.model_dump()`

**Warning:**
```
PydanticDeprecatedSince20: The `dict` method is deprecated; use `model_dump` instead.
```

**Cause:**
`web/app.py` called `profile.dict()` on a Pydantic v2 model. The `.dict()` method still works but is deprecated and will be removed in v3.

**Fix:**
```python
# Before
session_id = create_session(profile.dict())
session["agent"].set_profile(profile.dict())

# After
session_id = create_session(profile.model_dump())
session["agent"].set_profile(profile.model_dump())
```

---

## 23. `/chat/audio` — HTTP 422 Unprocessable Entity

**Error:**
```
Audio error: HTTP 422
```

**Cause:**
`session_id` in `web/app.py` was declared as a plain `str` parameter alongside an `UploadFile`. FastAPI interprets plain `str` parameters as query parameters. The frontend sent `session_id` as a multipart form field, so FastAPI could not find it → 422.

**Fix:**
Declare `session_id` explicitly as a form field using `Form(...)`:

```python
from fastapi import Form

async def chat_audio(
    session_id: str = Form(...),    # ← was: session_id: str
    audio: UploadFile = File(...),
):
```

Also updated the frontend to send `session_id` as a form field (not a query param) and save the audio as `.webm` matching the browser's `MediaRecorder` output format.

---

## 24. Recommendation API — `Connection refused` (service not running)

**Symptom:**
Avatar responded: *"I'm sorry, there's a temporary issue with our learning service."*

**Cause:**
The `recommend_courses` tool calls `http://localhost:8001/recommend`. That service was never started — it's an external API to be built by the software engineering team. `requests.post()` got `Connection refused`, the tool caught it and returned a service error string, which the LLM rephrased politely.

**Fix:**
Created `mock_services.py` — a FastAPI stub that runs on port 8001 and serves realistic course recommendations and assessments for demo and development use:

```bash
python mock_services.py   # starts mock APIs on port 8001
```

When the real LMS APIs are ready, point to them via environment variables:

```bash
export RECOMMENDATION_API_URL=https://your-lms.com/api/recommend
export ASSESSMENT_API_URL=https://your-lms.com/api/generate
```

---

## 25. XTTS and Whisper Hardcoded to CPU — GPU Laptop Gets No Speedup

**Problem:**
`voice/voice.py` had `self.device = "cpu"` hardcoded. `transcriber/transcriber.py` had `device="cpu"` hardcoded. Cloning the repo to an NVIDIA GPU laptop gave no GPU speedup.

**Cause:**
The CPU was originally forced to work around `aten::_fft_r2c` not being implemented on Apple MPS (issue #13). But this also blocked NVIDIA CUDA, which fully supports the op.

**Fix:**
Both files now auto-detect the best available device at startup:

`voice/voice.py`:
```python
if torch.cuda.is_available():
    self.device = "cuda"
else:
    self.device = "cpu"   # covers macOS MPS and CPU-only machines
```

`transcriber/transcriber.py`:
```python
if torch.cuda.is_available():
    device, compute_type = "cuda", "float16"
else:
    device, compute_type = "cpu", "int8"
```

Wav2Lip (subprocess) and Ollama already auto-detect CUDA — no changes needed there. The same codebase now runs optimally on both macOS (CPU) and NVIDIA GPU laptops without any manual changes.

**Expected speedup on NVIDIA GPU:**

| Stage | CPU | NVIDIA GPU |
|---|---|---|
| Ollama LLM | ~30–40s | ~3–6s |
| XTTS v2 | ~60–90s | ~4–8s |
| Wav2Lip | ~30–60s | ~2–5s |
| **Total** | **2–4 min** | **~10–20s** |

---

## 26. Text Reply and Lip-Sync Video Not Appearing at the Same Time

**Problem:**
After trying an async approach (return text immediately, generate video in background), the avatar kept showing a "thinking" ring for 2+ minutes after the text appeared. This was poor UX — the CEO demo needed text and video to arrive together.

**Root Cause:**
The async background task (FastAPI `BackgroundTasks`) returned the text reply immediately but the video wasn't ready for 2+ minutes. The frontend polled for it, but the long wait felt broken.

**Fix:**
Reverted to a **synchronous pipeline**. The API blocks until all three stages complete (LLM → XTTS → Wav2Lip) and returns text + video URL in a single response. Both appear in the browser at the same instant.

```python
def _run_avatar_pipeline(agent, text: str) -> tuple[str, str]:
    reply      = agent.run(text)          # LLM
    voice.synthesize(reply, temp_voice)   # XTTS
    lipsync.generate(..., temp_voice, video_path)  # Wav2Lip
    return reply, video_path
```

The tradeoff is a longer wait before anything appears (~2–4 min on CPU, ~10–20s on GPU), but the UX is cleaner — nothing appears until it's all ready, eliminating the confusing gap between text and video.

---

## 27. Frontend Demo — Avatar Idle State and CORS

**Changes made for the CEO demo frontend (`frontend/`):**

1. **CORS not configured** — `web/app.py` had no CORS headers. Browser blocked all requests from `localhost:3000` to `localhost:8000`.
   - Fix: Added `CORSMiddleware` with `allow_origins=["*"]` to `app.py`.

2. **Avatar image not served** — The frontend couldn't load `assets/hr_avatar.jpg`.
   - Fix: Added `StaticFiles` mount at `/assets` in `app.py`.

3. **Idle avatar had CSS spinner rings** — User requested the idle state show the actual avatar face video instead of animated rings.
   - Fix: Replaced rings with a looping `<video>` element playing `hr_avatar_silent.mp4`. This gives natural facial movement while the user is typing or waiting.

4. **Avatar states**:
   - `idle` → silent face video loops (natural movement)
   - `thinking` → same video + "Thinking…" label + blue border
   - `speaking` → lip-sync video plays + green border → returns to `idle` on end

---

## 28. Welcome Greeting — No Audio or Video Playing

**Symptom:**
On sign-in the text welcome message appeared in the chat but the avatar was silent and no video played.

**Cause:**
The welcome message was a static string hard-coded in the frontend JavaScript — no backend call was made. `playAvatarVideo()` was also using `.play()` directly without waiting for the video to buffer, causing the browser's autoplay policy to silently block it. By the time `/chat` was first called (~2–4 min later), the browser had closed the user-gesture window and blocked unmuted video autoplay.

**Fix (backend):**
Added `POST /session/welcome` to `web/app.py`. Builds a personalised greeting from the session profile, runs XTTS + Wav2Lip (no LLM call, so it's faster), and returns `reply` + `video_url` like a normal chat response.

```python
@app.post("/session/welcome", response_model=ChatResponse)
def session_welcome(request: WelcomeRequest):
    session = _require_session(request.session_id)
    profile = session["profile"]
    greeting = f"Welcome {name}! ..."
    voice.synthesize(greeting, output_path=temp_voice)
    lipsync.generate(AVATAR_SILENT_VIDEO, temp_voice, video_path)
    return ChatResponse(session_id=..., reply=greeting, video_url=...)
```

**Fix (frontend — autoplay):**
Changed `playAvatarVideo()` to listen for `oncanplay` before calling `.play()`, and added a click-to-play overlay for when the browser still blocks it:

```javascript
avatarVideo.oncanplay = () => {
  avatarVideo.play().catch(() => {
    playOverlay.classList.remove('hidden');  // user clicks to unblock
  });
};
```

**Fix (frontend — welcome call):**
Added `fetchWelcome(sid)` called immediately after login. The welcome plays in the browser the same as any chat response.

---

## 29. Browser-Side Voice Activity Detection

**Problem:**
The mic button required the user to manually click to start and click again to stop recording. This was clunky compared to the Python `vad/vad.py` behaviour, which automatically detects speech start and stop from continuous audio monitoring.

**Cause:**
The original frontend used `MediaRecorder` with explicit start/stop triggered by button clicks, with no amplitude analysis.

**Fix:**
Replaced manual start/stop with a browser-side VAD using the Web Audio API, mirroring the Python Silero VAD logic:

| Python (`vad.py`) | Browser (`app.js`) |
|---|---|
| Silero VAD model → speech probability | `AnalyserNode` → RMS amplitude |
| `speech_threshold = 0.5` | `SPEECH_THRESHOLD = 0.018` |
| `SILENCE_LIMIT = 20` frames (~640ms) | `SILENCE_LIMIT = 42` frames @ 60fps (~700ms) |
| `PyAudio` + `torch.hub` | `AudioContext` + `requestAnimationFrame` |
| Thread loop, puts to Queue | `requestAnimationFrame` loop, calls `sendAudio(blob)` |

**Implementation:**
- `startVAD()` — gets mic stream, creates `AudioContext` + `AnalyserNode`, starts `requestAnimationFrame` loop
- `monitorAudio()` — computes RMS each frame; calls `onSpeechStart()` / `onSpeechEnd()` based on threshold + silence counter
- `onSpeechStart()` — creates `MediaRecorder` on the existing stream, starts recording
- `onSpeechEnd()` — stops recorder; blob passed to existing `sendAudio(blob)`
- VAD auto-starts after the welcome video completes
- During backend processing (`isProcessing = true`) the loop keeps running but `onSpeechStart()` is blocked — no accidental triggers
- Mic button now toggles VAD on/off (mute/unmute) instead of start/stop recording

**Mic button states:**
- Grey (`vad-off`) — VAD off / muted
- Green pulse (`vad-listening`) — listening for speech
- Red pulse (`vad-speaking`) — speech detected, recording in progress

---

## 30. Azure Blob Storage — RAG Document Source

**Requirement:**
HR policy documents are stored in Azure Blob Storage. The RAG system needed to download and ingest them alongside local `hr_docs/` files.

**Fix:**
Extended `brain/rag.py` with two new methods:

- `ingest_from_azure(container_name, connection_string)` — connects via SDK, lists blobs, downloads `.pdf`/`.txt`/`.docx` to a temp dir, calls existing `ingest_documents()`
- `ingest_all(local_path, azure_container, azure_connection_string)` — convenience wrapper that runs both sources and returns total chunk count

Added `POST /admin/ingest` endpoint to `web/app.py` (Bearer-auth protected) so ingestion can be triggered via HTTP after deploying new documents.

Added `.docx` support to `ingest_documents()` using `Docx2txtLoader`.

**Required env vars:**
```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
export AZURE_STORAGE_CONTAINER="hr-documents"
```

**Dependencies added to `requirements.txt`:** `azure-storage-blob`, `docx2txt`

---

## 31. Wav2Lip Reloads Model From Disk on Every Request (~5–7s Wasted)

**Problem:**
`face/face.py` ran Wav2Lip via `subprocess.run(["python", "inference.py", ...])`. Each call spawned a fresh Python process that:
1. Imported all of Wav2Lip's dependencies (~2s)
2. Loaded the 416MB GAN checkpoint from disk (~3–5s)
3. Loaded the RetinaFace face detection model (~1s)
4. Ran inference
5. Exited — all models discarded

On the next request this entire sequence repeated. ~5–7s was wasted on model loading alone, every single turn.

**Fix:**
Import `inference.py` as a Python module at server startup, call `do_load()` once to load both models into module globals, then call `main()` directly on each request with a patched `args` namespace:

```python
# __init__: load once
import inference as _inf
_inf.do_load(self.checkpoint)   # Wav2Lip GAN + RetinaFace → module globals
self._inference = _inf

# generate(): call directly, no subprocess
self._inference.args = types.SimpleNamespace(wav2lip_batch_size=128, ...)
self._inference.main()
```

`inference.py` already separates model loading (`do_load()`) from inference (`main()`), and uses a module-level `args` namespace — making it straightforward to patch without modifying the upstream file.

**Saving:** ~5–7s per request eliminated. Combined with `beam_size=1` and GPU batch sizes, total pipeline drops from ~18–22s to ~8–12s on T4.

---

## 32. Running on Azure ML Compute Instance (Standard_NC8as_T4_v3)

See the `## Option C — Azure ML Compute Instance` section in `README.md` for the full step-by-step runbook.

**Key differences vs local Ubuntu setup:**
- CUDA drivers and toolkit are pre-installed — no manual CUDA install needed
- User is `azureuser`, home dir is `/home/azureuser/` (persists across restarts)
- No display — frontend is accessed via SSH port forwarding or VS Code port forwarding, not `localhost` directly
- Ollama must be installed manually (not pre-installed on Azure ML)
- `frontend/app.js` API constant must point to the instance's forwarded port, not hardcoded `localhost:8000`

**Port access pattern:**
```
Local browser → SSH tunnel → Azure ML instance:8000 (FastAPI)
Local browser → SSH tunnel → Azure ML instance:3000 (Frontend)
```

**Verified GPU check before running:**
```bash
nvidia-smi                                          # should show T4
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  Tesla T4
```

---

## Final Working Install Sequence

### macOS (Apple Silicon — CPU)

```bash
# 1. System deps
brew install pkg-config ffmpeg

# 2. Python environment
python3.10 -m venv hr_venv
source hr_venv/bin/activate

# 3. Python deps
pip install -r requirements.txt

# 4. TTS (bypass numpy pin)
pip install TTS==0.22.0 --no-deps
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk "ollama>=0.3.0" "langchain-ollama==0.1.3" \
    setuptools

# 5. Clone Wav2Lip
git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip

# 6. Download model checkpoints
curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
    -o wav2lip_gan.pth
curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth

# 7. Pull Ollama models
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

---

### Ubuntu / Windows — NVIDIA GPU (Recommended for production and demos)

```bash
# 1. System deps (Ubuntu)
sudo apt update && sudo apt install -y python3.10 python3.10-venv \
    ffmpeg pkg-config portaudio19-dev git curl build-essential libsndfile1

# Install CUDA 12.1+ from https://developer.nvidia.com/cuda-downloads
# Verify GPU is visible:
nvidia-smi

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 3. Clone repo and create environment
git clone https://github.com/abiolaks/hr_avatar.git
cd hr_avatar
python3.10 -m venv hr_venv
source hr_venv/bin/activate

# 4. Install PyTorch with CUDA FIRST (before requirements.txt)
#    This ensures the CUDA-enabled build is used, not the CPU-only default.
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Verify GPU is available to PyTorch
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: CUDA: True  <your GPU name>

# 6. Install remaining Python deps
pip install -r requirements.txt

# 7. TTS (bypass numpy pin)
pip install TTS==0.22.0 --no-deps
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk "ollama>=0.3.0" "langchain-ollama==0.1.3" \
    setuptools

# 8. Clone Wav2Lip
git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip

# 9. Download model checkpoints
curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
    -o wav2lip_gan.pth
curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth

# 10. Pull Ollama models (Ollama auto-uses GPU)
ollama pull qwen3:4b
ollama pull nomic-embed-text

# No code changes needed — XTTS, Whisper, Wav2Lip, and Ollama all auto-detect CUDA.
```

---

---

## 33. Wav2Lip — CUDA Out of Memory (Competing Ollama Process)

**Error:**
```
CUDA out of memory. Tried to allocate 1.53 GiB. GPU 0 has a total capacity of 11.50 GiB
of which 1.24 GiB is free. Process 8240 has 4.57 GiB memory in use.
```

**Cause:**
Ollama keeps the LLM loaded in GPU VRAM for 5 minutes after each response (its default `keep_alive`). When Wav2Lip runs immediately after the LLM step, both models compete for GPU memory. With an 11.5 GiB GPU: ~5.7 GiB (our process) + 4.57 GiB (Ollama) = ~10.3 GiB in use, leaving only ~1.2 GiB free — not enough for Wav2Lip's peak allocation.

**Fix — two changes:**

1. **Set `keep_alive=0` in `ChatOllama`** (`brain/agent.py`) — tells Ollama to unload the model from GPU immediately after generating a response, freeing the 4.57 GiB before TTS/Wav2Lip run:
```python
self.llm = ChatOllama(
    model=OLLAMA_MODEL,
    ...
    keep_alive=0,   # unload from GPU immediately — frees VRAM for Wav2Lip
)
```

2. **Reduce `wav2lip_batch_size` from 32 to 8** (`face/face.py`) — lowers peak VRAM during the inference forward pass as a safety margin:
```python
wav2lip_batch_size = 8,
```

3. **Add `torch.cuda.empty_cache()`** before and after inference in `face/face.py` — releases fragmented allocations between requests.

**Trade-off:** Ollama reloads the model on each request (~2–3s), but this is absorbed into the TTS synthesis time and invisible to the user.

---

## 34. Mistral Tool-Call JSON Leaking at End of Response

**Symptom:**
Avatar replied with visible raw JSON at the end of its message:
```
To test your knowledge, let's generate an assessment for the Python Basics course.
[{"name":"generate_assessment","arguments":{"course_id":"Python Basics"}}]
```

**Cause:**
Mistral (via Ollama) sometimes emits tool invocations as plain text content rather than structured tool calls. The existing regex in `brain/agent.py` only stripped the `[TOOL_CALLS]` prefix at the start of the message, not trailing JSON arrays at the end.

**Fix:**
Added a second regex to strip trailing tool-call JSON arrays:
```python
# Before: only stripped leading [TOOL_CALLS] token
ai_message = re.sub(r'^\[TOOL_CALLS\]\s*(\[.*?\])?\s*', '', ai_message, flags=re.DOTALL)

# After: also strips trailing [{"name":...}] arrays
ai_message = re.sub(r'^\[TOOL_CALLS\]\s*(\[.*?\])?\s*', '', ai_message, flags=re.DOTALL)
ai_message = re.sub(r'\s*\[\s*\{\s*"name"\s*:.*?\}\s*\]\s*$', '', ai_message, flags=re.DOTALL)
```

---

## 35. Assessment API — Wrong Port (8002 vs 8001)

**Symptom:**
Assessment tool not working — `generate_assessment` tool calls failed silently or the agent emitted raw tool-call JSON.

**Cause:**
`ASSESSMENT_API_URL` in `config.py` was hardcoded to `http://localhost:8002/generate`. But `mock_services.py` serves both `/recommend` and `/generate` on port **8001**. Port 8002 was never started.

**Fix:**
```python
# config.py — before
ASSESSMENT_API_URL = os.getenv("ASSESSMENT_API_URL", "http://localhost:8002/generate")

# config.py — after
ASSESSMENT_API_URL = os.getenv("ASSESSMENT_API_URL", "http://localhost:8001/generate")
```

---

## 36. Lip-Sync and TTS Cut Off Mid-Sentence

**Symptom:**
Avatar would speak and display a full answer but stop before the last sentence — e.g. a leave policy answer ended at "20 working days." and cut "If you have any questions about other." entirely.

**Cause:**
`num_predict=150` in `ChatOllama` was too low for multi-point policy answers. The model hit the token limit mid-sentence. The truncated text was passed directly to XTTS and Wav2Lip, which faithfully synthesised the incomplete response.

**Fix — two changes:**

1. **Increase `num_predict` from 150 to 300** (`brain/agent.py`):
```python
num_predict=300,  # enough for policy summaries
```

2. **Add `_trim_to_last_sentence()` safety net** (`brain/agent.py`) — if the model still hits the limit mid-sentence, the dangling fragment is dropped before TTS ever sees it:
```python
def _trim_to_last_sentence(text: str) -> str:
    if re.search(r'[.!?]["\']?\s*$', text):
        return text
    match = re.search(r'^(.*[.!?]["\']?)\s+\S', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
```
Called at the end of the response cleanup block in `HRAgent.run()`.

---

### Running after setup (both platforms)

```bash
# Terminal 1 — LLM
ollama serve

# Terminal 2 — Avatar API (auto-detects GPU if available)
source hr_venv/bin/activate
uvicorn web.app:app --host 0.0.0.0 --port 8000

# Terminal 3 — Mock LMS services (dev/demo only)
source hr_venv/bin/activate
python mock_services.py

# Terminal 4 — Frontend demo
cd frontend && python -m http.server 3000
# Open http://localhost:3000
```

---

## 37. Agent Returns "retrieve_policy" as Text Instead of Policy Content

**Symptom:**
When an employee asked a policy question (e.g. "What is the annual leave policy?"), the avatar spoke and displayed the literal text `retrieve_policy` instead of the actual policy answer. Logs showed:
```
INFO - Agent response: retrieve_policy...
```
No `"Tool: retrieve_policy called"` log entry appeared between the user input and the response.

**Cause:**
The original `brain/agent.py` used `langgraph.prebuilt.create_react_agent` to drive the conversation loop. With `granite4:3b`, the ReAct loop did not reliably execute the tool call — it returned the tool name as plain text content instead of a structured `tool_calls` object, and the loop exited without running the tool. The raw tool name was then passed through to TTS and Wav2Lip unchanged.

**Fix:**
Replaced `create_react_agent` with a custom `HRAgent` class that handles tool calls explicitly in a single pass:

1. **Structured tool call path** (`response.tool_calls` populated): model returns a proper function-call object → tool is executed directly.
2. **Leaked-tool-call fallback** (`response.content` is a bare tool name or JSON): detected by `_try_execute_leaked_tool_call()` → tool is executed using the user's original message as the query argument.
3. **Policy summarisation**: `retrieve_policy` results are multi-paragraph, so a second LLM call (with no tools bound) summarises them to 1–3 sentences for TTS delivery.

This eliminated the runaway-loop problem (granite4 was chaining 6 tool calls in the old ReAct setup, ~57s latency) and guarantees the tool always executes.

**Files changed:** `brain/agent.py` (full rewrite)

---

## 38. Assessment Output Truncated — Last Question Missing Options

**Symptom:**
When the avatar generated an assessment, the last question was spoken and displayed without its answer options:
```
Would you recommend this course to a colleague?
```
The options (Definitely / Probably / Not sure / No) were silently dropped.

**Cause:**
`_trim_to_last_sentence()` was applied unconditionally to every agent response — including direct tool outputs. The function searches for the last `[.!?]` followed by a non-whitespace character and trims everything after it. For assessment text, the last question ends with `?` and is followed by answer options that have no terminal punctuation. The regex matched the `?` at the end of the question and trimmed the trailing options as a "dangling fragment".

**Root pattern:**
```
Would you recommend this course to a colleague?   ← ? found here
Definitely                                         ← trimmed (no punctuation before EOF)
Probably
Not sure
No
```

**Fix:**
Added a `_needs_sentence_trim` flag (default `True`) that is set to `False` for all paths where the response is complete tool output rather than LLM-generated prose:

```python
_needs_sentence_trim = True   # LLM prose may be cut off mid-sentence

# After _phrase_tool_result() for assessments / recommendations:
_needs_sentence_trim = False  # tool output is already complete

# Only trim LLM-generated content
if _needs_sentence_trim:
    ai_message = _trim_to_last_sentence(ai_message)
```

`_trim_to_last_sentence` still runs for:
- Policy summaries (LLM-generated, may be cut off by `num_predict`)
- Direct LLM answers with no tool call

**File changed:** `brain/agent.py`

---

## 39. `vectorstore.persist()` Deprecation Warning on Every Ingestion

**Symptom:**
Every call to `RAGManager._add_documents()` (triggered by `ingest_documents`, `ingest_from_azure`, or `ingest_all`) printed a deprecation warning to stderr:
```
LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is
no longer supported as docs are automatically persisted.
```

**Cause:**
`brain/rag.py` called `self.vectorstore.persist()` after adding documents. In `chromadb >= 0.4.0` (the version in `requirements.txt`), persistence is automatic — every `add_documents` call writes to disk immediately. The explicit `persist()` call is a no-op but emits a `LangChainDeprecationWarning` on every invocation.

**Fix:**
Removed the `self.vectorstore.persist()` call. Documents are persisted automatically by chromadb 0.4.x with no action required:

```python
# Before
self.vectorstore.add_documents(chunks)
self.vectorstore.persist()   # ← removed
logger.info(f"Ingested {len(chunks)} chunks into vectorstore")

# After
self.vectorstore.add_documents(chunks)
logger.info(f"Ingested {len(chunks)} chunks into vectorstore")
```

**File changed:** `brain/rag.py`

---

## 40. OllamaEmbeddings Replaced with FastEmbedEmbeddings

**Symptom / Motivation:**
RAG retrieval added ~4.6 seconds of latency per turn because `OllamaEmbeddings` made an HTTP round-trip to the Ollama server for every query. On a policy question this meant:

```
User message → embed query (4.6s) → ChromaDB search → LLM summarise
```

Additionally, if Ollama was not running or the `nomic-embed-text` model was not pulled, RAG would silently fail with a connection error.

**Cause / Background:**
The original `brain/rag.py` used:
```python
from langchain_community.embeddings import OllamaEmbeddings
self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)  # nomic-embed-text
```
This required Ollama to be running and the embedding model pre-pulled.

**Fix:**
Switched to `FastEmbedEmbeddings`, which runs the embedding model in-process (no HTTP, no Ollama dependency):
```python
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

Results:
- Embedding latency: **4.6s → ~0.05s** per query
- No Ollama dependency for RAG — embeddings work even if Ollama is down
- `nomic-embed-text` no longer needs to be pulled

**Important:** The vectorstore was re-ingested after this change. `BAAI/bge-small-en-v1.5` produces 384-dimensional vectors vs. 768 from `nomic-embed-text`. Mixing embedding models in one ChromaDB collection causes incorrect similarity scores — if switching models on an existing vectorstore, always delete `chroma_db/` and re-ingest:

```bash
rm -rf chroma_db/
curl -X POST http://localhost:8000/admin/ingest \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"local_path": "./hr_docs"}'
```

**Files changed:** `brain/rag.py`, `config.py` (removed `EMBEDDING_MODEL` import from rag.py)

---

## 41. Assessment Catalog — Generic Fallback for All Courses

**Symptom / Motivation:**
Any course the employee mentioned that wasn't `python-101` or `ml-101` returned the same generic 3-question self-reflection set:
```
What was the main concept covered in '[course]'?
How confident are you in applying what you learned?
Would you recommend this course to a colleague?
```
This was unhelpful for knowledge testing and did not reflect the actual course content.

**Cause:**
`mock_services.py` only had two entries in `ASSESSMENTS` — `python-101` and `ml-101`.

**Fix:**
Added real subject-specific questions for all 24 courses in the recommendation catalog. Each entry has 3 multiple-choice questions with 4 options, keyed by the course name lowercased with spaces replaced by hyphens (matching how the agent extracts and passes the `course_id`):

| Category | New IDs added |
|---|---|
| Python | `cs50p`, `kaggle-python`, `automate-the-boring-stuff`, `python-for-everybody`, `python-oop` |
| Machine Learning | `google-ml-crash-course`, `kaggle-intro-to-machine-learning`, `microsoft-ml-for-beginners`, `machine-learning-specialization`, `kaggle-intermediate-machine-learning`, `fastai-practical-machine-learning`, `stanford-cs229` |
| Deep Learning | `deep-learning-specialization`, `mit-intro-deep-learning`, `fastai-deep-learning-foundations` |
| AI Agents | `introduction-to-ai-agents`, `ai-agents-in-langgraph`, `langchain-for-llm-application-development`, `functions-tools-and-agents-with-langchain`, `building-agentic-rag-with-llamaindex` |
| Data Science | `kaggle-pandas`, `kaggle-data-visualisation`, `data-analysis-with-python`, `kaggle-feature-engineering` |

Total assessments: 26 (24 course-specific + 2 legacy IDs `python-101`, `ml-101` kept for backwards compatibility).

**File changed:** `mock_services.py`

---

## 43. `_video_jobs` Dictionary — Unbounded Memory Growth (`web/app.py`)

**Symptom:** Server memory usage grows steadily over time with no upper bound.

**Cause:**
Every `/chat` and `/chat/audio` call adds a job entry to `_video_jobs`. The dict was never cleaned up — jobs accumulated indefinitely regardless of whether the client ever polled them.

**Fix — two-layer cleanup:**

1. **Immediate cleanup at poll time** — `GET /video/status/{job_id}` now deletes the entry as soon as it returns `ready` or `error`. This covers the normal case (client always polls).

2. **Periodic background sweep** — A `_prune_video_jobs()` function removes jobs older than 10 minutes. This covers the edge case where the client disconnects without polling. It runs every 5 minutes inside the lifespan background task (see fix #45).

```python
# web/app.py — created_at timestamp added to each job
_video_jobs[job_id] = {"status": "pending", "created_at": time.time()}

# Immediate cleanup in video_status()
_video_jobs.pop(job_id, None)

# Background sweep
def _prune_video_jobs(max_age_seconds: int = 600) -> None:
    cutoff = time.time() - max_age_seconds
    stale = [jid for jid, j in _video_jobs.items() if j.get("created_at", 0) < cutoff]
    for jid in stale:
        _video_jobs.pop(jid, None)
```

**Files changed:** `web/app.py`

---

## 44. `datetime.utcnow()` Deprecated in Python 3.12+ (`brain/session.py`)

**Symptom:** `DeprecationWarning: datetime.utcnow() is deprecated` on Python 3.12+. Will raise in a future Python version.

**Cause:** `session.py` used `datetime.utcnow()` (returns a naive datetime with no timezone) for all session timestamps. `logger.py` already used the correct `datetime.now(timezone.utc)` — `session.py` did not.

**Fix:** Import `timezone` and replace all three occurrences:
```python
# Before
from datetime import datetime, timedelta
datetime.utcnow()

# After
from datetime import datetime, timedelta, timezone
datetime.now(timezone.utc)
```

**Files changed:** `brain/session.py`

---

## 45. `@app.on_event("startup")` and `asyncio.get_event_loop()` Deprecated (`web/app.py`)

**Symptom:** FastAPI logs a deprecation warning for `@app.on_event`; Python 3.10+ logs a deprecation warning for `asyncio.get_event_loop()` called inside a running async context.

**Cause:**
- `@app.on_event("startup")` was deprecated in FastAPI in favour of the `lifespan` context manager pattern.
- `asyncio.get_event_loop()` is deprecated for obtaining the *running* loop inside an async function; the correct call is `asyncio.get_running_loop()`.

**Fix:** Migrate to `lifespan` and fix the loop call:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    loop = asyncio.get_running_loop()   # correct inside async context
    await loop.run_in_executor(None, _render_welcome_video)
    prune_task = asyncio.create_task(_prune_loop())
    yield
    # shutdown
    prune_task.cancel()
    try:
        await prune_task
    except asyncio.CancelledError:
        pass

app = FastAPI(..., lifespan=lifespan)
```

The `lifespan` pattern also hosts the periodic `_prune_video_jobs` + `prune_expired_sessions` background task (see fixes #43 and #46).

**Files changed:** `web/app.py`

---

## 46. Session Store — No Proactive Expiry (`brain/session.py`)

**Symptom:** Abandoned sessions (employee closes tab without clicking logout) stay in the `_store` dict forever. On a multi-user deployment this is a slow memory leak.

**Cause:** Expiry was lazy — a session was only removed when it was next accessed and found to be stale. Sessions that were never accessed again were never cleaned up.

**Fix:** Added `prune_expired_sessions()` to `brain/session.py`. It scans `_store` and removes all sessions inactive longer than `SESSION_TTL_MINUTES`. It is called every 5 minutes by the lifespan background task in `web/app.py`.

```python
def prune_expired_sessions() -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
    expired = [sid for sid, s in _store.items() if s["last_active"] < cutoff]
    for sid in expired:
        _store.pop(sid, None)
    return len(expired)
```

**Files changed:** `brain/session.py`, `web/app.py`

---

## 47. Dockerfile — Missing C Compiler for pip Packages

**Symptom:** Docker build fails or silently skips packages like `cython`, `pyaudio`, `gruut` that compile C extensions during `pip install`.

**Cause:** The `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` base image does not include `gcc`/`g++`. The `build-essential` meta-package was missing from the `apt-get install` list.

**Fix:** Add `build-essential` to the Dockerfile:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ...
```

**Files changed:** `Dockerfile`
