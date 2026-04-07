# HR Avatar — Handover Deliverables

This document defines what the AI Engineer is handing over and what each engineering team must build to take the system to production.

---

## What the AI Engineer Has Built

The AI engineer owns the **complete AI pipeline** — everything from receiving text/audio to returning a lip-synced video response. The system is production-ready at the AI layer.

### Delivered Components

| Component | File(s) | Status |
|---|---|---|
| LangGraph ReAct agent (profile-aware, tool-calling) | `brain/agent.py` | Done |
| RAG manager — ChromaDB + OllamaEmbeddings | `brain/rag.py` | Done |
| HR policy tools (search, recommend, assess) | `brain/tools.py` | Done |
| Session store + LMS profile injection via ContextVar | `brain/session.py`, `brain/session_context.py` | Done |
| Speech-to-text — faster-whisper (CUDA auto-detected) | `transcriber/transcriber.py` | Done |
| Voice synthesis — Coqui XTTS v2 (voice cloning) | `voice/voice.py` | Done |
| Lip-sync — Wav2Lip GAN (in-process, model loaded once) | `face/face.py` | Done |
| Voice activity detection — Silero VAD | `vad/vad.py` | Done |
| FastAPI web server with all LMS integration endpoints | `web/app.py` | Done |
| Browser-side VAD (mirrors Python VAD in JS) | `frontend/app.js` | Done |
| Demo frontend UI (CEO-ready, single-page app) | `frontend/` | Done |
| Mock recommendation + assessment APIs | `mock_services.py` | Done (dev/demo only) |
| Azure Blob Storage RAG ingestion + DOCX support | `brain/rag.py` | Done |
| Unit + integration tests | `tests/` | Done |
| HR policy document (ingested into ChromaDB) | `hr_docs/` | Done |
| Full setup runbooks (macOS, Ubuntu GPU, Azure ML, Docker) | `README.md` | Done |
| Blockers log with all 36 issues and fixes | `blockers_n_solutions.md` | Done |

### API Surface Delivered

The AI engineer exposes these endpoints for the engineering teams to integrate against:

| Method | Path | Caller | Purpose |
|---|---|---|---|
| `POST` | `/session/start` | LMS backend | Start session, inject employee profile |
| `POST` | `/session/welcome` | LMS frontend | Personalised spoken welcome video (no LLM) |
| `POST` | `/chat` | LMS frontend | Text conversation turn → reply + video_job_id |
| `POST` | `/chat/audio` | LMS frontend | Audio upload → transcribe → reply + video_job_id |
| `GET` | `/video/status/{job_id}` | LMS frontend | Poll for lip-sync video readiness |
| `GET` | `/video/{id}` | LMS frontend | Stream generated lip-sync video |
| `DELETE` | `/session/{id}` | LMS backend | End session on logout |
| `GET` | `/health` | LMS / monitoring | Liveness check |
| `POST` | `/admin/ingest` | Admin / CI | Trigger RAG ingestion (local + Azure Blob) |

Full request/response schemas and example payloads are in `README.md → LMS Integration`.

---

## Backend Engineer — What You Must Build

The Avatar calls two external APIs that must be built by the backend team. Currently `mock_services.py` stubs them both on port 8001.

### 1. Course Recommendation API

**Called by:** `brain/tools.py → recommend_courses()`

**Endpoint:**
```
POST <RECOMMENDATION_API_URL>     # env var, default: http://localhost:8001/recommend
```

**Request body the Avatar sends:**
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
  "preferred_category": "machine learning"
}
```

**Expected response:**
```json
{
  "recommendations": [
    {
      "title": "Machine Learning Fundamentals",
      "description": "Intro to ML with Python — covers regression, classification, and clustering.",
      "difficulty": "Intermediate",
      "duration": "Short",
      "url": "https://lms.company.com/courses/ml-fundamentals"
    }
  ]
}
```

The Avatar formats these into a spoken, conversational response. Any fields beyond `title` and `description` are optional but improve personalisation.

### 2. Assessment Generation API

**Called by:** `brain/tools.py → generate_assessment()`

**Endpoint:**
```
POST <ASSESSMENT_API_URL>     # env var, default: http://localhost:8001/generate
```

**Request body:**
```json
{
  "course_id": "ml-basics"
}
```

**Expected response:**
```json
{
  "assessment": [
    {
      "question": "What is the bias-variance trade-off?",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "B"
    }
  ]
}
```

The Avatar reads the questions aloud and presents them in the chat as a spoken quiz.

### 3. Production Session Store

Currently sessions are held **in-memory** in `brain/session.py`. They are lost on server restart and cannot be shared across multiple Avatar instances.

**Replace with:** Redis (or any persistent key-value store).

The session schema is a plain Python dict:
```python
{
  "agent": HRAgent(),          # LangGraph agent instance (not serialisable — must be rebuilt on restore)
  "profile": { ...UserProfile fields... },
  "last_active": datetime
}
```

Agent instances are not serialisable. On cache miss (restart/expiry), rebuild the agent from the stored profile dict and restore conversation history from a persisted message log.

### 4. Video Storage

Lip-sync videos are written to `/tmp` and served directly. They are lost on restart and not suitable for multi-instance deployments.

**Replace with:** Azure Blob Storage, S3, or a persistent volume.

The handover point is in `web/app.py → _lipsync_job()`: after `lipsync.generate()` completes, upload the file to object storage and return a CDN/signed URL instead of the `/video/{id}` local path.

### 5. Production Authentication

`/session/start` and `/admin/ingest` are protected by a shared secret (`LMS_SHARED_SECRET`). This is sufficient for an internal LMS-to-Avatar server call.

For production, consider:
- Rotating the shared secret (current: env var, never commit)
- Adding per-user JWT validation if the Avatar is ever called directly from the browser
- Rate limiting `/chat` and `/chat/audio` per session

### 6. Environment Variables to Set in Production

```bash
LMS_SHARED_SECRET=<strong-random-secret>
RECOMMENDATION_API_URL=https://your-lms.com/api/recommend
ASSESSMENT_API_URL=https://your-lms.com/api/generate
AZURE_STORAGE_CONNECTION_STRING=<connection-string>
AZURE_STORAGE_CONTAINER=hr-documents
OLLAMA_HOST=http://localhost:11434   # or container service URL
```

---

## Frontend Engineer — What You Must Build

The demo UI in `frontend/` is a standalone proof-of-concept. It is **not** designed to be embedded into the LMS as-is. The frontend engineer must build the production LMS widget that integrates with the Avatar API.

### 1. Session Lifecycle Integration

The LMS frontend must:

1. **On widget open:** call `POST /session/start` from the **LMS backend** (not the browser — the backend holds `LMS_SHARED_SECRET`). Store `session_id` in the LMS frontend state.
2. **Immediately after session start:** call `POST /session/welcome` from the browser with `session_id`. Play the returned `video_url` to greet the employee.
3. **On widget close / logout:** call `DELETE /session/{session_id}` to clean up.

### 2. Chat Turn Flow

Each conversation turn follows this pattern:

```
User types or speaks
        ↓
POST /chat  (text)  or  POST /chat/audio  (microphone blob)
        ↓
Response: { reply, video_job_id }
        ↓
Display reply text immediately
        ↓
Poll GET /video/status/{video_job_id}  every 1s
        ↓
When ready=true: play video_url in avatar video element
```

The reference implementation is in `frontend/app.js`. Key functions:
- `sendMessage(text)` — text path
- `sendAudio(blob)` — audio path
- `pollVideoStatus(jobId)` — polling loop
- `playAvatarVideo(url)` — plays video, returns to idle on end

### 3. Voice Activity Detection

The browser-side VAD in `frontend/app.js` is production-quality and can be used as-is. It uses the Web Audio API (`AnalyserNode` + RMS amplitude), mirrors the Python Silero VAD behaviour, and handles the full mic state machine (idle → listening → recording → processing).

**Key constants to tune per deployment:**
```javascript
const SPEECH_THRESHOLD = 0.018;   // amplitude threshold — raise in noisy environments
const SILENCE_LIMIT    = 42;      // frames of silence before stopping (~700ms)
```

### 4. Avatar Video States

The avatar element should implement three states (see `frontend/style.css` and `frontend/app.js` for reference):

| State | Visual | Trigger |
|---|---|---|
| `idle` | Silent face video loops | Default / after video ends |
| `thinking` | Same video + loading indicator | After sending message, before video is ready |
| `speaking` | Lip-sync video plays | When `video_url` is returned and video is ready |

The silent face video (`assets/hr_avatar_silent.mp4`) must be available at `/assets/hr_avatar_silent.mp4` on the Avatar API server.

### 5. Autoplay Handling

Browsers block unmuted video autoplay without a prior user gesture. The demo handles this with a click-to-play overlay (`frontend/app.js → playAvatarVideo()`). The production LMS integration should ensure the welcome video triggers within the user's login gesture window (i.e., call `/session/welcome` immediately after the user clicks "Open Avatar" — not after a delay).

### 6. Audio Format

`MediaRecorder` in Chrome/Firefox defaults to `.webm` (Opus codec). The Avatar server accepts this — `transcriber/transcriber.py` uses `faster-whisper` which handles `.webm` via ffmpeg. Do not convert to WAV in the browser; send the raw `MediaRecorder` blob.

---

## Mock Services Reference

`mock_services.py` provides realistic stub responses for dev and demo. It serves both APIs on **port 8001**:

- `POST /recommend` — returns 3 courses filtered by `preferred_difficulty` and `preferred_duration`
- `POST /generate` — returns 2-question assessments for `python-basics`, `ml-basics`, `data-analysis`, `sql-advanced`

To run:
```bash
source hr_venv/bin/activate
python mock_services.py
```

Point the real APIs via env vars when ready:
```bash
export RECOMMENDATION_API_URL=https://your-lms.com/api/recommend
export ASSESSMENT_API_URL=https://your-lms.com/api/generate
```

---

## Known Limitations to Address in Production

| Limitation | Impact | Resolution |
|---|---|---|
| In-memory session store | Sessions lost on restart; no horizontal scaling | Replace with Redis (backend team) |
| Videos stored in `/tmp` | Lost on restart; no CDN | Upload to Azure Blob / S3 after generation (backend team) |
| `mock_services.py` | Recommendation and assessment data are fake | Build real APIs (backend team) |
| Shared secret auth only | No per-user token validation | Add JWT if Avatar is ever called directly from browser (backend team) |
| `allow_origins=["*"]` CORS | Too permissive for production | Lock down to LMS domain (backend team) |
| Single-worker lipsync executor | One video generated at a time | Scale horizontally with a job queue (backend team) |
| Ollama `keep_alive=0` | Model reloads on every request (~2–3s overhead) | Acceptable on T4; revisit if latency SLA tightens |

---

## Contact / Questions

All setup issues, design decisions, and architectural trade-offs are documented in:
- `README.md` — full setup runbooks and API reference
- `blockers_n_solutions.md` — 36 issues encountered and resolved, with exact fixes
