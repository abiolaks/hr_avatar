# HR Avatar — Complete Technical Implementation Reference

> **Target audience:** A senior engineer reading this codebase for the first time who needs to fully understand, reproduce, debug, and extend the system.
>
> **Coverage:** Every component, every logic path, every non-obvious line of code. Nothing is omitted.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Full Architecture Diagram](#2-full-architecture-diagram)
3. [Component Deep-Dives](#3-component-deep-dives)
   - 3.1 [config.py](#31-configpy)
   - 3.2 [logger.py](#32-loggerpy)
   - 3.3 [brain/session_context.py](#33-brainsession_contextpy)
   - 3.4 [brain/session.py](#34-brainsessionpy)
   - 3.5 [brain/rag.py — RAGManager](#35-brainragpy--ragmanager)
   - 3.6 [brain/tools.py](#36-braintoolspy)
   - 3.7 [brain/agent.py — HRAgent](#37-brainagentpy--hragent)
   - 3.8 [transcriber/transcriber.py](#38-transcribertranscriberpy)
   - 3.9 [voice/voice.py — VoiceSynthesizer](#39-voicevoicepy--voicesynthesizer)
   - 3.10 [face/face.py — LipSyncGenerator](#310-facefacepy--lipsyncgenerator)
   - 3.11 [web/app.py — FastAPI Server](#311-webapppy--fastapi-server)
   - 3.12 [mock_services.py](#312-mock_servicespy)
   - 3.13 [frontend/app.js](#313-frontendappjs)
4. [The Agent Turn — Full Walkthrough](#4-the-agent-turn--full-walkthrough)
5. [RAG Pipeline](#5-rag-pipeline)
6. [Tool Execution Logic — All Three Paths](#6-tool-execution-logic--all-three-paths)
7. [Session Lifecycle](#7-session-lifecycle)
8. [GPU Memory Management](#8-gpu-memory-management)
9. [Frontend ↔ Backend Protocol](#9-frontend--backend-protocol)
10. [Data Flow Diagrams — The Three Tool Types](#10-data-flow-diagrams--the-three-tool-types)
11. [Design Decisions and Trade-offs](#11-design-decisions-and-trade-offs)
12. [Reproduction Checklist](#12-reproduction-checklist)

---

## 1. System Overview

HR Avatar is an AI-powered conversational assistant embedded in a company Learning Management System (LMS). An employee opens a widget, logs in with their profile, and can talk to an animated HR avatar — either by typing or speaking — to ask about HR policies, request course recommendations, or get assessed on a course they completed.

**What the employee sees:** A video of the HR avatar whose lips move in sync with a cloned HR voice, answering their question. Text appears in a chat bubble simultaneously, word by word, timed to the video duration.

**What happens under the hood, end to end:**

1. The LMS backend creates a session by posting the employee's profile (name, job role, department, skill level, known skills, enrolled courses) to the FastAPI backend.
2. The employee's question arrives as text (typed) or audio (microphone). If audio, Whisper transcribes it.
3. The text goes to the HRAgent, which sends it to a local Ollama LLM (IBM Granite 4, 3B parameters) with three tools bound to it.
4. The LLM either calls a tool or answers directly. Tools are: `retrieve_policy` (RAG over company HR documents), `recommend_courses` (calls an external LMS API), or `generate_assessment` (calls an external LMS API).
5. The tool result or LLM answer is cleaned up for text-to-speech — all markdown stripped.
6. XTTS v2 synthesizes audio in the cloned HR voice.
7. Wav2Lip GAN animates a static avatar image to lip-sync with the audio, producing an MP4.
8. The frontend polls for the video, plays it, and streams the text word by word in sync.

**Critical constraints the architecture is built around:**
- A single GPU with limited VRAM (~6–8 GB). Whisper (large-v3, ~3.8 GB), XTTS (~2 GB), and Wav2Lip cannot all be resident simultaneously without OOM errors.
- TTS responses must be short enough for natural dialogue. Long answers cause 40+ second waits and VRAM OOM.
- The LLM must never invent HR policy or course names — all factual answers must be grounded in tool results.

---

## 2. Full Architecture Diagram

```
EMPLOYEE BROWSER
┌─────────────────────────────────────────────────────────────────────┐
│  index.html + style.css                                              │
│  app.js                                                              │
│                                                                      │
│  ┌──────────────┐    ┌───────────────────────────────────────────┐  │
│  │  Login Form  │    │  Chat Screen                               │  │
│  │  (profile    │    │  ┌─────────────┐  ┌─────────────────────┐ │  │
│  │   fields)    │    │  │ Avatar Video│  │ Chat Bubbles        │ │  │
│  └──────┬───────┘    │  │ (Wav2Lip    │  │ (word-stream sync'd │ │  │
│         │            │  │  output MP4)│  │  to video duration) │ │  │
│         │            │  └─────────────┘  └─────────────────────┘ │  │
│         │            │  ┌──────────────────────────────────────┐  │  │
│         │            │  │ Text input + Send btn                │  │  │
│         │            │  │ Mic btn → Silero VAD (WASM)          │  │  │
│         │            │  │   onSpeechEnd → float32ToWav blob    │  │  │
│         │            │  └──────────────────────────────────────┘  │  │
│         │            └───────────────────────────────────────────┘  │
└────┬────┴──────────────────────────────────────┬────────────────────┘
     │  POST /session/start                      │  POST /chat
     │  Authorization: Bearer dev-secret         │  POST /chat/audio (FormData)
     │  Body: UserProfile JSON                   │  GET /video/status/{job_id}  (poll)
     │                                           │  GET /video/{video_id}       (stream)
     ▼                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  web/app.py — FastAPI server (port 8000)                            │
│                                                                      │
│  Startup:                                                            │
│    Transcriber()  ← WhisperModel large-v3 loaded into GPU           │
│    VoiceSynthesizer() ← XTTS v2 loaded, speaker latents pre-cached  │
│    LipSyncGenerator() ← Wav2Lip GAN + RetinaFace loaded into GPU    │
│    _render_welcome_video() ← pre-renders assets/welcome.mp4         │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ POST /session/start                                            │  │
│  │   validate Bearer token → create_session() → HRAgent()        │  │
│  │   agent.set_profile(profile) → sets context var               │  │
│  │   returns {session_id}                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ POST /session/welcome                                          │  │
│  │   returns pre-rendered welcome.mp4 immediately                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ POST /chat                          POST /chat/audio           │  │
│  │   agent.run(text) ──────────────────────────────────────────┐ │  │
│  │                                     Transcriber.transcribe() │ │  │
│  │                                     → agent.run(text) ───────┘ │  │
│  │   reply text returned immediately                              │  │
│  │   _start_lipsync_async(reply) → job_id                        │  │
│  │     └→ ThreadPoolExecutor (max_workers=1)                      │  │
│  │          VoiceSynthesizer.synthesize() → /tmp/voice_*.wav      │  │
│  │          LipSyncGenerator.generate()  → /tmp/output_*.mp4     │  │
│  │   returns {reply, video_job_id}                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ GET /video/status/{job_id}  → {ready, video_url}              │  │
│  │ GET /video/{video_id}       → HTTP 206 Range-capable stream   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│  brain/agent.py          │   │  GPU (shared VRAM budget)            │
│  HRAgent                 │   │                                      │
│                          │   │  Startup residents:                  │
│  ChatOllama(             │   │    XTTS v2            ~2.0 GB        │
│    model=granite4:3b,    │   │    Wav2Lip GAN        ~0.5 GB        │
│    temperature=0.3,      │   │    RetinaFace         ~0.1 GB        │
│    num_ctx=2048,         │   │                                      │
│    num_predict=200,      │   │  Transcription slot (sequential):    │
│    keep_alive=0,         │   │    Whisper large-v3   ~3.8 GB        │
│  )                       │   │    (loaded → infer → UNLOADED)       │
│  .bind_tools(tools)      │   │                                      │
│                          │   │  LLM slot (Ollama managed):          │
│  run(user_input):        │   │    granite4:3b        ~2.5 GB        │
│    1. build messages[]   │   │    keep_alive=0 → freed immediately  │
│    2. llm_with_tools     │   │    after each response               │
│       .invoke(msgs)      │   │                                      │
│    3. check 3 paths:     │   └──────────────────────────────────────┘
│       leaked tool call   │
│       structured tool    │   ┌──────────────────────────────────────┐
│       direct answer      │   │  Ollama (localhost:11434)            │
│    4. clean output       │   │    model: granite4:3b                │
│    5. event_logger.log() │   │    keep_alive=0 → unloads model      │
│    returns answer text   │   │    from VRAM after each call         │
└──────────┬───────────────┘   └──────────────────────────────────────┘
           │
     ┌─────┴──────────────────────────────────────┐
     │                                            │
     ▼                                            ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│  brain/tools.py          │   │  brain/rag.py — RAGManager           │
│                          │   │                                      │
│  retrieve_policy(query)  │   │  FastEmbedEmbeddings                 │
│    → rag.retrieve(q, k=5)│   │    model: BAAI/bge-small-en-v1.5     │
│    → top-5 chunks        │   │    in-process, no HTTP round-trip    │
│    → format as string    │   │                                      │
│                          │   │  Chroma vectorstore                  │
│  recommend_courses(...)  │   │    persist_dir: ./chroma_db/         │
│    → profile from ctx var│   │                                      │
│    → POST :8001/recommend│   │  Ingestion: local ./hr_docs/         │
│    → top-3 courses       │   │             + Azure Blob Storage     │
│                          │   │    chunk_size=300, overlap=100       │
│  generate_assessment(id) │   │    RecursiveCharacterTextSplitter    │
│    → POST :8001/generate │   │                                      │
│    → questions list      │   │  Retrieval: similarity_search(q, k=5)│
└──────────────────────────┘   └──────────────────────────────────────┘
                                             ▲
                               ┌─────────────┘
                               │  retrieve_policy calls rag.retrieve()
                               │
┌──────────────────────────────────────────────────────────────────────┐
│  mock_services.py (port 8001)                                        │
│                                                                      │
│  POST /recommend — keyword scoring over COURSES catalog              │
│    filter by difficulty + enrolled_courses exclusion                 │
│    sort by keyword relevance score                                   │
│    return top 3                                                      │
│                                                                      │
│  POST /generate — ASSESSMENTS dict lookup by course_id              │
│    fallback: 3 generic reflection questions                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep-Dives

### 3.1 `config.py`

**What it does:** Central configuration — computes absolute paths and reads environment variables. Imported by almost every other module.

**Key constants and their effects:**

| Constant | Value / Source | Effect if changed |
|---|---|---|
| `BASE_DIR` | `dirname(abspath(__file__))` | Anchor for all relative paths. If you move the project, this auto-adjusts. |
| `ASSETS_DIR` | `BASE_DIR/assets` | Where `hr_avatar.jpg`, `hr_voice_sample.wav`, `hr_avatar_silent.mp4`, and the pre-rendered `welcome.mp4` live. |
| `CHROMA_DIR` | `BASE_DIR/chroma_db` | ChromaDB persistence directory. Delete this to force re-ingestion. |
| `LOGS_DIR` | `BASE_DIR/logs` | Log files: `hr_avatar.log` (structured text) and `events.jsonl` (per-turn events). |
| `VOICE_SAMPLE` | `assets/hr_voice_sample.wav` | Reference audio for XTTS speaker cloning. The voice the avatar speaks in. Must be 6–30 seconds of clean speech. |
| `AVATAR_IMAGE` | `assets/hr_avatar.jpg` | Used by Wav2Lip as the face source. |
| `AVATAR_SILENT_VIDEO` | `assets/hr_avatar_silent.mp4` | Pre-rendered silent looping video of the avatar. Wav2Lip uses this as the face input rather than a still image, producing smoother head motion. |
| `OLLAMA_MODEL` | `granite4:3b` | LLM used by HRAgent. Change this to `llama3.1`, `mistral`, etc. — but the leaked-tool-call handler in agent.py must cover that model's quirks. |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Unused by RAGManager (which uses FastEmbed). Legacy field. |
| `RECOMMENDATION_API_URL` | env `RECOMMENDATION_API_URL` or `http://localhost:8001/recommend` | Where `recommend_courses` tool sends its POST. Swap to the real LMS API in production. |
| `ASSESSMENT_API_URL` | env `ASSESSMENT_API_URL` or `http://localhost:8001/generate` | Where `generate_assessment` tool sends its POST. |
| `LMS_SHARED_SECRET` | env `LMS_SHARED_SECRET` or `dev-secret` | Bearer token the LMS must send to create sessions and trigger ingestion. Change in production. |
| `AZURE_STORAGE_CONNECTION_STRING` | env var | Connection string for Azure Blob Storage ingestion. Empty = disabled. |
| `AZURE_STORAGE_CONTAINER` | env or `hr-documents` | Azure container name. |
| `LOG_LEVEL` | env `LOG_LEVEL` or `INFO` | Console log verbosity. Set to `DEBUG` to see every document loaded, every chunk, every tool call. |

`os.makedirs(..., exist_ok=True)` for `ASSETS_DIR`, `CHROMA_DIR`, and `LOGS_DIR` runs at import time, so those directories always exist when any other module tries to write to them.

---

### 3.2 `logger.py`

**What it does:** Sets up two loggers and one structured event sink.

**`logger`** — a standard Python `logging.Logger` named `"hr_avatar"`. Writes to:
- `logs/hr_avatar.log` (file, DEBUG and above)
- stdout (console, INFO and above)

**`EventLogger`** — writes one JSON line per conversation turn to `logs/events.jsonl`. Each record contains:

```json
{
  "tool_called": "retrieve_policy",
  "tool_args": {"query": "annual leave"},
  "hallucination_guard": false,
  "user_id": "emp_001",
  "input": "What is the annual leave policy?",
  "response": "You are entitled to 25 days...",
  "latency_ms": 2341,
  "grounded": true,
  "timestamp": "2026-04-02T10:23:01.123Z"
}
```

Parse with `pandas.read_json('logs/events.jsonl', lines=True)` or `jq` for evaluation.

**`log_performance`** — a function decorator. Wraps any function and logs `PERF | funcname took X.XXXs` to the console. Applied to `HRAgent.run`, `RAGManager.retrieve`, `RAGManager.ingest_documents`, `Transcriber.transcribe`, `VoiceSynthesizer.synthesize`, and `LipSyncGenerator.generate`.

```python
def log_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"PERF | {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper
```

The `@wraps(func)` preserves the original function's `__name__` and `__doc__` — important because LangChain reads the docstring of `@tool`-decorated functions to generate the tool schema. Without `@wraps`, wrapping a tool with `@log_performance` would lose the docstring and break tool binding.

---

### 3.3 `brain/session_context.py`

**What it does:** Thread-safe, request-scoped storage for the active user's LMS profile. This allows the tool functions (which are module-level, not instance methods) to access the current user's profile without passing it as an argument.

```python
from contextvars import ContextVar

_session_profile: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "session_profile", default=None
)
```

**Why `ContextVar` and not a global variable:** FastAPI runs async handlers in the same thread pool and can multiplex multiple requests. A plain global dict keyed by session_id would work but is awkward. Python's `contextvars.ContextVar` provides per-task (per-async-task or per-thread) storage — each concurrent request has its own copy. This is the same mechanism Django and Starlette use for request-local storage.

**API:**
- `set_profile(profile: dict)` — called by `HRAgent.run()` at the start of every turn and by `HRAgent.set_profile()` at session creation.
- `get_profile()` — called by `recommend_courses` tool to build the full payload for the external API.

**Why called twice per turn:** `HRAgent.set_profile()` is called once when the session is created (to set `self._profile`). `set_profile` from `session_context` is called inside `HRAgent.run()` on every turn because the Python context variable may not have survived from the session creation call (depends on the async executor). Belt and suspenders.

---

### 3.4 `brain/session.py`

**What it does:** In-memory session store. Maps `session_id → {profile, agent, created_at, last_active}`.

**`create_session(profile)`:**
```python
session_id = f"sess_{uuid.uuid4().hex[:16]}"
_store[session_id] = {
    "profile": profile,
    "agent": HRAgent(),       # a dedicated agent instance per session
    "created_at": datetime.utcnow(),
    "last_active": datetime.utcnow(),
}
```
Each session gets its own `HRAgent` instance, which owns its own `self.messages` conversation history. This isolation means two simultaneous users never share context.

**`get_session(session_id)`:**
- Returns `None` for unknown IDs.
- Checks `last_active < now - 60 minutes`. If stale, calls `delete_session()` and returns `None`. This causes the FastAPI handler to raise HTTP 404.
- Updates `last_active = now` on every access, implementing sliding-window TTL.

**`delete_session(session_id)`:** Called explicitly when the employee clicks "End Session" (DELETE `/session/{id}`) and automatically on TTL expiry.

**`active_session_count()`:** Returns `len(_store)`. Exposed via `GET /health`.

**Design note — no persistence:** Sessions live in RAM only. A server restart loses all sessions. This is intentional for the current architecture — adding Redis or a database is the natural next step for production.

---

### 3.5 `brain/rag.py` — RAGManager

**What it does:** Manages the ChromaDB vector store used for HR policy retrieval. Handles ingestion (file loading → chunking → embedding → storage) and retrieval (query → embedding → similarity search → top-k chunks).

#### Embedding model choice

```python
self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

`FastEmbedEmbeddings` (from `langchain-community`) runs the `BAAI/bge-small-en-v1.5` model **in-process** using ONNX Runtime. There is no HTTP round-trip. This is the most significant performance decision in the RAG layer. The alternative, `OllamaEmbeddings`, sends each embedding request to the Ollama HTTP server — adding ~4.6 seconds per document chunk at ingestion time and ~0.5 seconds per query at retrieval time. With hundreds of chunks, `OllamaEmbeddings` would make ingestion take minutes instead of seconds.

`BAAI/bge-small-en-v1.5` is a 33M parameter sentence embedding model. It produces 384-dimensional vectors and is competitive with much larger models on MTEB benchmarks for retrieval tasks.

#### Chunking parameters

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
```

- `chunk_size=300` characters: Short chunks fit entirely in the LLM's context window (2048 tokens) even when five chunks are concatenated. A 300-character chunk is approximately 50–75 tokens.
- `chunk_overlap=100` characters: Ensures sentences that span chunk boundaries are captured in at least one chunk. Without overlap, a sentence starting at character 299 would be split, potentially losing context.

`RecursiveCharacterTextSplitter` tries to split on `\n\n`, then `\n`, then `. `, then ` `, then individual characters. This produces semantically coherent chunks (paragraph → sentence → word) rather than arbitrary character cuts.

#### Ingestion paths

**Local directory (`ingest_documents`):**
```python
for glob, loader_cls in [
    ("**/*.txt",  TextLoader),
    ("**/*.pdf",  PyPDFLoader),
    ("**/*.docx", Docx2txtLoader),
]:
    loader = DirectoryLoader(docs_path, glob=glob, loader_cls=loader_cls)
    docs = loader.load()
```
`DirectoryLoader` recurses through subdirectories. Each file is loaded into a list of `Document` objects (LangChain's data container: `page_content` + `metadata`).

**Azure Blob Storage (`ingest_from_azure`):**
1. Create `BlobServiceClient` from connection string.
2. List all blobs in the container.
3. Skip blobs with unsupported extensions (not `.txt`, `.pdf`, `.docx`).
4. Download each supported blob to a temporary directory, flattening folder slashes to `__` in the filename to avoid creating subdirectories.
5. Call `ingest_documents(tmp_dir)` on the temp directory — reusing the same pipeline.
6. The `TemporaryDirectory` context manager deletes all downloaded files when done.

**`ingest_all`:** Convenience method that calls both local and Azure ingestion in sequence. Returns the total chunk count. Called by `POST /admin/ingest`.

#### Retrieval

```python
def retrieve(self, query: str, k: int = 3) -> list:
    docs = self.vectorstore.similarity_search(query, k=k)
    return docs
```

Called by `retrieve_policy` with `k=5`. ChromaDB embeds `query` using the same `FastEmbedEmbeddings` model (so the embedding space is consistent), computes cosine similarity against all stored chunk vectors, and returns the top 5 `Document` objects. The tool concatenates `doc.page_content` for all returned docs and returns the combined string to the agent.

---

### 3.6 `brain/tools.py`

**What it does:** Defines the three LangChain tools available to the HRAgent. LangChain's `@tool` decorator reads the function's name, docstring, and type annotations to generate an OpenAI-compatible tool schema that is sent to the LLM as part of the system prompt.

#### `retrieve_policy(query: str) -> str`

```python
docs = rag.retrieve(query, k=5)
context = "\n\n".join([doc.page_content for doc in docs])
return f"Relevant policy information:\n{context}"
```

- Retrieves 5 chunks (more than the default 3 in `RAGManager.retrieve`) to give the LLM more context for policy questions, which are often spread across multiple paragraphs.
- Returns "No relevant policy documents found." if nothing matches — the agent treats this as a signal to tell the employee to contact HR.
- The `rag` instance at the module level is created once at import time, so ChromaDB is only opened once per process.

#### `recommend_courses(learning_goal, preferred_difficulty, preferred_duration, preferred_category) -> str`

The docstring is deliberately long and detailed — it tells the LLM exactly how to fill each parameter, including how to infer duration from time mentions. This docstring is part of the tool schema sent to the LLM.

```python
profile = get_profile() or {}
payload = {
    "user_id":              profile.get("user_id", ""),
    "job_role":             profile.get("job_role", ""),
    ...
    "learning_goal":        learning_goal,
    "preferred_difficulty": preferred_difficulty or profile.get("skill_level", "Beginner"),
    ...
}
response = requests.post(RECOMMENDATION_API_URL, json=payload, timeout=10)
```

Key points:
- `get_profile()` reads from the context variable set by `HRAgent.run()`. This lets tools access per-user data without passing it explicitly.
- `preferred_difficulty or profile.get("skill_level")` — if the employee didn't state a difficulty preference, fall back to their skill level from the LMS profile. The LLM is instructed to do this mapping, but the tool also does it as a safety net.
- A 10-second timeout prevents one slow external API call from hanging the entire request.
- The result is formatted as markdown: `[Title](url): Description\n` per course. The agent's `_phrase_tool_result` strips the markdown for TTS.

#### `generate_assessment(course_id: str = None) -> str`

```python
payload = {"course_id": course_id}
response = requests.post(ASSESSMENT_API_URL, json=payload, timeout=10)
questions = data.get("questions", [])
result = "Here is your assessment:\n"
for i, q in enumerate(questions, 1):
    result += f"{i}. {q['question']}\n"
    if "options" in q:
        for opt in q["options"]:
            result += f"   - {opt}\n"
```

Produces a numbered list. `_phrase_tool_result` in the agent strips the numbers and dashes for TTS readability.

**Tool export:**
```python
tools = [retrieve_policy, recommend_courses, generate_assessment]
```
This list is imported in `agent.py` and passed to `self.llm.bind_tools(tools)`.

---

### 3.7 `brain/agent.py` — HRAgent

This is the core intelligence of the system. Read this section carefully — it contains the most non-obvious logic.

#### Initialization

```python
self.llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3,
    num_ctx=2048,
    num_predict=200,
    keep_alive=0,
)
self.llm_with_tools = self.llm.bind_tools(tools)
```

- `temperature=0.3`: Lower than the default 0.8. More deterministic, less creative. For an HR assistant answering policy questions, creativity is undesirable.
- `num_ctx=2048`: Context window size in tokens. The default for Ollama is model-dependent (often 4096). 2048 is sufficient for the system prompt (~400 tokens) + 10 turns of conversation (~1000 tokens) + tool schemas (~300 tokens). Reducing this saves memory and speeds up attention computation.
- `num_predict=200`: Hard cap on output tokens. This is the most consequential parameter for latency and reliability: (a) prevents runaway responses that would take 40+ seconds to synthesize; (b) XTTS will OOM on very long audio; (c) the lipsync video grows proportionally to audio length. 200 tokens ≈ 150 words ≈ 2–3 sentences, which is the target response length.
- `keep_alive=0`: Ollama's model residence time. Setting this to 0 tells Ollama to unload the model from VRAM **immediately** after each response is complete. This frees ~2.5 GB that TTS and Wav2Lip need during the next step. Without this, Ollama holds the model in GPU RAM indefinitely, causing OOM when XTTS or Wav2Lip try to run.

#### System prompt

The `_BASE_SYSTEM_PROMPT` is 500+ characters and does several things:

1. **Role definition:** "HR assistant embedded in the LMS."
2. **Tool mandate:** For each of the three responsibility types, it says "ALWAYS call [tool]" and "Never answer from memory." This combats the model's tendency to answer from its training data.
3. **Tool parameter guidance:** For `recommend_courses`, it explains exactly how to map user language to parameter values ("beginner" → "Beginner", time mentions → duration category).
4. **Response format rules:** "Plain text only. No markdown. Maximum 3 sentences." — These rules are enforced again programmatically via regex in `run()`, but having them in the prompt reduces how often the cleanup is needed.
5. **Clarification rule:** If the input is garbage (transcription noise), ask for clarification instead of guessing.
6. **No tool narration:** "Never say 'I will use the X tool'." Without this, models often announce their actions.

`_build_system_prompt(profile)` appends a profile block:
```
Employee profile (from LMS — do not ask the employee for these):
- Name: Abiola K.
- Job role: Data Analyst
- Department: Engineering
- Skill level: Intermediate
- Known skills: SQL, Python
- Enrolled courses: none
```
This personalization is injected at call time on every turn, not at session creation, so `set_profile()` doesn't need to invalidate any cached state.

#### `run(user_input: str) -> str` — the three paths

The method appends the user input to `self.messages`, builds the full message list (system + history), and makes a single LLM call:

```python
response = self.llm_with_tools.invoke(msgs)
ai_message = response.content or ""
```

Then it evaluates three mutually exclusive conditions:

**Path 1 — Leaked tool call:**
```python
tool_result = _try_execute_leaked_tool_call(ai_message, user_input)
if tool_result:
    ai_message = _phrase_tool_result(self.llm, tool_result)
    _needs_sentence_trim = False
```
Some models (granite4, llama3.1) occasionally output the tool call as plain text inside `response.content` instead of structured `response.tool_calls`. The helper `_try_execute_leaked_tool_call` detects two forms:
- **Bare tool name:** The model outputs just `retrieve_policy` as its entire response. The function checks if `stripped in _TOOLS_MAP`.
- **Raw JSON:** The model outputs `{"name": "retrieve_policy", "parameters": {"query": "annual leave"}}`. The function parses this, extracts `name` and `parameters`/`arguments`, and invokes the tool.

**Path 2 — Structured tool call (normal path):**
```python
elif response.tool_calls:
    tool_call = response.tool_calls[0]  # first only — extras silently ignored
    clean_args = {k: v for k, v in tool_call["args"].items() if v is not None}
    raw_result = _TOOLS_MAP[tool_call["name"]].invoke(clean_args)
```

For `retrieve_policy` only, there is a second LLM call to summarize:
```python
if tool_call["name"] == "retrieve_policy":
    follow_up = msgs + [
        response,
        ToolMessage(content=str(raw_result), tool_call_id=tool_call["id"]),
    ]
    summary = self.llm.invoke(follow_up)  # self.llm, NOT llm_with_tools
    ai_message = summary.content
```
Why `self.llm` and not `self.llm_with_tools`? Because `self.llm_with_tools` has tools bound, meaning a model could call another tool in response to the policy result. Using `self.llm` (no tools bound) prevents this — it can only output text. This is the explicit "hard stop" preventing chained tool calls.

For `recommend_courses` and `generate_assessment`, the raw tool result is processed directly via `_phrase_tool_result()` without a second LLM call. These results are already structured lists — the LLM would add nothing useful, and a second call would waste 2+ seconds.

**Path 3 — Direct answer:**
```python
else:
    _course_openers = ("here are some", "here's some", "recommended course",
                       "i recommend", "you might want to")
    if any(p in ai_message.lower() for p in _course_openers):
        # Hallucinated course list detected
        result = _TOOLS_MAP["recommend_courses"].invoke({"learning_goal": user_input})
        ai_message = _phrase_tool_result(self.llm, result)
```
The hallucination guard catches the case where the model answers a course question from its own training knowledge instead of calling the tool. It detects course-list-shaped responses by matching opener phrases and forces the correct tool call. This is logged as `"hallucination_guard": true` in the event log.

For truly direct answers (greetings, clarification requests, off-topic deflections), the model's response passes through after markdown stripping and sentence trimming.

#### Post-processing pipeline

After the three paths, every response goes through the same cleanup:

```python
# 1. Strip Mistral's leaked tool-call tokens
ai_message = re.sub(r'^\[TOOL_CALLS\]\s*(\[.*?\])?\s*', '', ai_message, flags=re.DOTALL)
ai_message = re.sub(r'\s*\[\s*\{\s*"name"\s*:.*?\}\s*\]\s*$', '', ai_message, flags=re.DOTALL)

# 2. Strip markdown (TTS reads symbols aloud)
ai_message = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', ai_message)   # **bold**, *italic*
ai_message = re.sub(r'#{1,6}\s*', '', ai_message)                  # ## headers
ai_message = re.sub(r'^\s*[-*]\s+', '', ai_message, flags=re.MULTILINE)  # - bullets
ai_message = re.sub(r'^\s*\d+\.\s+', '', ai_message, flags=re.MULTILINE) # 1. numbered

# 3. Strip tool narration phrases
ai_message = re.sub(
    r"^.*?\bI (will|am going to|'ll) use the ['\"]?\w+['\"]? tool\.?\s*",
    '', ai_message, flags=re.IGNORECASE | re.DOTALL,
)

# 4. Sentence trim (only for LLM prose, not tool outputs)
if _needs_sentence_trim:
    ai_message = _trim_to_last_sentence(ai_message)
```

`_trim_to_last_sentence(text)`: If `num_predict=200` cuts the response mid-sentence, this function finds the last `.`, `!`, or `?` and trims to that point. Without this, TTS might speak a sentence fragment like "The annual leave policy allows employees to take up to 25 days, which can be". The flag `_needs_sentence_trim` is `False` for tool outputs (which are structured lists and always "complete") and `True` for LLM-generated prose.

---

### 3.8 `transcriber/transcriber.py`

**What it does:** Converts audio to text using `faster-whisper` (CTranslate2-optimized Whisper).

#### Audio preprocessing

```python
def _to_wav(audio_path: str) -> tuple[str, bool]:
    result = subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000",   # Whisper's native sample rate
        "-ac", "1",       # mono
        "-f", "wav",
        wav_path,
    ], capture_output=True)
```

Whisper requires 16 kHz mono WAV. The frontend VAD (`@ricky0123/vad-web`) already produces Float32 audio at 16 kHz and the JS `float32ToWav()` encodes it as a proper WAV file, so this conversion is a safety net for other input formats (e.g., `.webm` from older browser recordings). `should_delete=True` means the temp file will be cleaned up.

#### Model parameters

```python
self.model_size = "large-v3"
self.compute_type = "float16" if self.device == "cuda" else "int8"
```

`large-v3` is the highest-accuracy open Whisper model. `float16` on CUDA is half-precision — approximately half the VRAM of `float32` with no significant accuracy loss. `int8` on CPU is quantized integer arithmetic — much slower but avoids GPU dependency.

#### Transcription parameters

```python
segments, _ = self.model.transcribe(
    wav_path,
    language="en",
    beam_size=5,
    temperature=0,
    condition_on_previous_text=False,
    vad_filter=False,
    initial_prompt=("HR assistant, company policy, learning management system, ...")
)
```

- `language="en"`: Skips language detection, saving ~0.3s per call.
- `beam_size=5`: Standard beam search width. Higher = more accurate but slower.
- `temperature=0`: Greedy decoding. No randomness — transcription should be deterministic.
- `condition_on_previous_text=False`: Don't use previous segments as context for the next. Prevents hallucination loops where Whisper generates text from a previous turn if silence is detected.
- `vad_filter=False`: Internal Whisper VAD is disabled because the browser-side Silero VAD already clips to speech segments. Enabling both would double-process.
- `initial_prompt`: A comma-separated list of domain-relevant words. Whisper uses this to bias its vocabulary toward HR/LMS terminology. Without this, it might transcribe "scikit-learn" as "Sci-kit learn" or miss "Pandas" vs "pandas".

#### Reliability filtering

```python
reliable = [
    seg for seg in segments
    if seg.avg_logprob > -1.0 and seg.no_speech_prob < 0.6
]
```

`avg_logprob` is Whisper's own confidence score (log probability). Below -1.0 means the model is very uncertain — typically noise or inaudible audio. `no_speech_prob > 0.6` means Whisper thinks the segment is more likely silence than speech. Filtering these out prevents garbage transcriptions like "mhm" or random phonemes from reaching the LLM.

#### GPU unloading after each transcription

```python
finally:
    self._unload()

def _unload(self):
    del self.model
    self.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

Whisper large-v3 occupies ~3.8 GB of VRAM. After transcription, this memory must be freed before XTTS and Wav2Lip can run. On the next audio request, `transcribe()` checks `if self.model is None` and calls `self._load()` to reload from the HuggingFace local cache (~3–5 seconds overhead). This load-use-unload cycle is the core strategy for fitting the pipeline on a single GPU with limited VRAM.

---

### 3.9 `voice/voice.py` — VoiceSynthesizer

**What it does:** Synthesizes speech from text in a cloned voice using Coqui XTTS v2 (a zero-shot multilingual TTS model).

#### Speaker conditioning latent caching

```python
tts_model = self.tts.synthesizer.tts_model
self._gpt_cond_latent, self._speaker_embedding = tts_model.get_conditioning_latents(
    audio_path=[speaker_wav_path],
    gpt_cond_len=30,
    gpt_cond_chunk_len=4,
    max_ref_length=60,
)
```

XTTS v2 is a GPT-based TTS model that uses a reference audio clip to condition the speaker's voice. Normally, calling `tts.tts(text, speaker_wav=...)` recomputes these conditioning latents from the WAV file on every call (~2–3 seconds). By calling `get_conditioning_latents()` once at startup and caching the result in `self._gpt_cond_latent` and `self._speaker_embedding`, each subsequent synthesis skips this step entirely.

#### Direct model inference

```python
out = tts_model.inference(
    text=text,
    language="en",
    gpt_cond_latent=self._gpt_cond_latent,
    speaker_embedding=self._speaker_embedding,
    temperature=0.7,
    enable_text_splitting=True,
)
```

`tts_model.inference()` is called directly instead of `self.tts.tts()`. The high-level `tts()` method always reloads speaker conditioning from the WAV file. The low-level `inference()` accepts pre-computed tensors. This is a deliberate bypass of the intended public API — it's the only way to achieve the latent caching optimization with the Coqui TTS library.

- `temperature=0.7`: Controls prosody variability. Higher = more expressive but less consistent. 0.7 is a good balance.
- `enable_text_splitting=True`: XTTS generates audio in chunks for long text. This allows it to handle paragraphs without VRAM OOM.

#### Text preprocessing

```python
def _clean_for_tts(text: str) -> str:
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)  # [label](url) → label
    text = re.sub(r"https?://\S+", "", text)               # bare URLs
    text = re.sub(r"(\*{1,3}|_{1,3})|`{1,3}[^`]*`{1,3}|...", " ", text)
    text = text.replace("!", ".")        # ! → . for natural cadence
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()
```

TTS models pronounce symbols literally: `*bold*` would be spoken as "asterisk bold asterisk". This cleanup runs before synthesis. The agent also strips markdown before returning, so this is a second layer of defense.

#### Hard length cap

```python
if len(text) > 600:
    cut = text.rfind('.', 0, 600)
    text = text[:cut + 1] if cut != -1 else text[:600]
```

At 600 characters (approximately 100 words or ~45 seconds of speech), XTTS starts taking very long (40+ seconds) and can OOM. The cap truncates at the last sentence boundary within 600 characters. This is a safety valve — the agent's `num_predict=200` token limit should prevent responses this long in the first place.

#### Output format

```python
sf.write(output_path, np.array(wav), samplerate=24000)
```

XTTS v2 outputs at 24 kHz. `soundfile.write` saves as WAV. This WAV is then passed to Wav2Lip.

---

### 3.10 `face/face.py` — LipSyncGenerator

**What it does:** Generates a lip-synced MP4 video of the avatar speaking, using Wav2Lip GAN and RetinaFace face detector.

#### One-time model loading

```python
def _load_models(self):
    if WAV2LIP_DIR not in sys.path:
        sys.path.insert(0, WAV2LIP_DIR)
    os.makedirs(os.path.join(WAV2LIP_DIR, "temp"), exist_ok=True)

    orig_dir = os.getcwd()
    os.chdir(WAV2LIP_DIR)
    try:
        import inference as _inf
        importlib.reload(_inf)
        _inf.do_load(self.checkpoint)  # loads both Wav2Lip GAN and RetinaFace
        self._inference = _inf
    finally:
        os.chdir(orig_dir)
```

`inference.py` in the `wav2lip/` directory is the original Wav2Lip inference script. It uses module-level globals for the loaded models and uses `args` as a module-level namespace. The strategy here is:
1. Add `wav2lip/` to `sys.path` so `import inference` resolves to the local script.
2. `importlib.reload(_inf)` ensures a clean module state (important if this runs more than once).
3. `_inf.do_load(checkpoint)` is a custom function that runs only the model-loading portion of inference — not defined in the original script. It loads the Wav2Lip GAN (from `wav2lip_gan.pth`) and the RetinaFace face detector, both into GPU memory, and stores them as module globals.
4. `os.chdir(WAV2LIP_DIR)` is required because `inference.py` resolves its `temp/` directory relative to `cwd`.

**Before this optimization:** Each `generate()` call spawned a subprocess running `python inference.py ...`, which loaded both models from disk on every request — adding 5–7 seconds overhead per video.

**After:** Models are loaded once at startup and held in `_inf`'s module globals. Each `generate()` call only runs the inference computation.

#### Per-request inference

```python
self._inference.args = types.SimpleNamespace(
    checkpoint_path    = self.checkpoint,
    face               = os.path.abspath(face_video_path),
    audio              = os.path.abspath(audio_path),
    outfile            = os.path.abspath(output_path),
    static             = False,
    fps                = 25.0,
    pads               = pads,
    wav2lip_batch_size = 8,
    resize_factor      = 2,    # 2x downscale before face detection — faster, same quality at screen res
    out_height         = 360,  # reduced from 480; cuts per-frame processing time
    crop               = [0, -1, 0, -1],
    box                = [-1, -1, -1, -1],
    rotate             = False,
    nosmooth           = True,   # skip temporal smoothing — saves ~15% per frame
    img_size           = 96,
)
os.chdir(WAV2LIP_DIR)
self._inference.main()
```

`inference.main()` is the original Wav2Lip processing pipeline. It:
1. Reads the `args` namespace (which the original script read from argparse — here injected directly).
2. Opens the face video (`hr_avatar_silent.mp4`), extracting frames.
3. Detects faces in each frame using RetinaFace.
4. Extracts the mel spectrogram from the audio WAV.
5. Feeds frame+mel pairs to the Wav2Lip GAN in batches of 8.
6. Writes the output MP4 to `outfile`.

**Why a silent video as the face source?** Using `hr_avatar_silent.mp4` (a looping video of the avatar with natural head movement) rather than a still `hr_avatar.jpg` produces smoother lip sync. Wav2Lip's temporal consistency works better with pre-existing motion than with a static image that gets repeated.

**Performance flags:**
- `resize_factor=2`: Downsamples the input video 2x before face detection. RetinaFace runs on smaller frames, saving ~30% time. The output is still upsampled to `out_height`.
- `out_height=360`: Down from the original 480. 360p is sufficient for a chat widget embed. Reduces per-frame GPU memory.
- `nosmooth=True`: Skips Wav2Lip's temporal smoothing (blending between adjacent frame face crops). This saves ~15% render time with no visible difference on the small output size.
- `wav2lip_batch_size=8`: Process 8 face-audio pairs per GPU forward pass. Higher = more VRAM but faster.

---

### 3.11 `web/app.py` — FastAPI Server

**What it does:** The HTTP bridge between the LMS frontend and every backend component. Orchestrates the full pipeline.

#### Startup

```python
transcriber = Transcriber()    # loads Whisper large-v3 into GPU
voice       = VoiceSynthesizer()  # loads XTTS v2 into GPU, pre-computes speaker latents
lipsync     = LipSyncGenerator()  # loads Wav2Lip GAN + RetinaFace into GPU
```

These are module-level singletons. All heavy models load once at startup. At startup, Whisper, XTTS, and Wav2Lip are all in GPU memory simultaneously. The VRAM budget is approximately: Whisper (3.8 GB) + XTTS (2.0 GB) + Wav2Lip+RetinaFace (0.6 GB) = 6.4 GB. On a GPU with 8 GB VRAM this is tight but workable. If the GPU has less VRAM, Whisper will fail to load.

The welcome video is pre-rendered at startup:
```python
@app.on_event("startup")
async def startup_tasks():
    if not os.path.exists(_WELCOME_VIDEO_PATH):
        await loop.run_in_executor(None, _render_welcome_video)
```
If `assets/welcome.mp4` already exists (from a previous run), it is reused without re-rendering. This means the first startup takes ~30–60 seconds (TTS + Wav2Lip for the welcome message), but subsequent restarts are fast. Deleting `assets/welcome.mp4` forces a re-render.

#### The video job queue

```python
_lipsync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_video_jobs: dict = {}
```

`max_workers=1` is the critical constraint. TTS and Wav2Lip share GPU memory. Running two synthesis jobs concurrently would OOM. The single-worker executor serializes all jobs into a FIFO queue. If two users send messages simultaneously, the second user's video job waits until the first is complete.

`_video_jobs` maps `job_id → {status, video_path, error}`. Status values: `"pending"` (submitted, not started), `"ready"` (complete), `"error"`.

#### `POST /chat`

```python
reply  = agent.run(request.message)   # synchronous — waits for LLM
job_id = _start_lipsync_async(reply)  # submits to executor, returns immediately
return ChatResponse(session_id=..., reply=reply, video_job_id=job_id)
```

The response is returned to the frontend immediately after the LLM answers, before TTS or Wav2Lip have started. This means the frontend gets the text reply in ~2–4 seconds (LLM latency only), then polls for the video separately.

#### `POST /chat/audio`

```python
temp_audio = f"/tmp/upload_{uuid.uuid4().hex}{extension}"
text = transcriber.transcribe(temp_audio)  # blocks until Whisper finishes + unloads
reply  = agent.run(text)
job_id = _start_lipsync_async(reply)
```

The audio path is longer: save upload → transcribe (Whisper) → LLM → async TTS+Wav2Lip. The endpoint is `async def` to allow FastAPI to handle other requests during `await audio.read()`, but `transcriber.transcribe()` is synchronous blocking code (runs in the default thread pool via uvicorn's thread pool).

#### `GET /video/status/{job_id}`

Simple dictionary lookup. Returns `{ready: false}` while pending, `{ready: true, video_url: "/video/output_xyz.mp4"}` when done, or `{ready: false, error: "..."}` on failure.

#### `GET /video/{video_id}`

Serves the generated MP4 from `/tmp/` with HTTP Range support:

```python
range_header = request.headers.get("range")
if range_header:
    # Parse "bytes=start-end"
    start = int(start_str) if start_str else 0
    end   = int(end_str)   if end_str   else file_size - 1
    return StreamingResponse(iterfile(), status_code=206, ...)
```

Range requests (HTTP 206) are required for browser `<video>` elements to seek and buffer video correctly. Without Range support, many browsers cannot play the video at all — they send an initial Range request and give up if they get a plain 200. The generator `iterfile()` reads in 64 KB chunks to avoid loading the entire MP4 into memory.

#### `POST /session/welcome`

Returns immediately with the pre-rendered `welcome.mp4` path:
```python
return ChatResponse(
    session_id=request.session_id,
    reply=_WELCOME_GREETING,
    video_url="/assets/welcome.mp4",
)
```
`video_url` (not `video_job_id`) means the frontend skips polling and plays the video directly. `/assets/` is a `StaticFiles` mount pointing at the `assets/` directory.

#### `POST /admin/ingest`

Protected by the same Bearer token. Creates a fresh `RAGManager()` and calls `ingest_all()`. Intended to be called once at deployment to load HR documents, and again whenever documents change. The RAGManager writes to the persisted ChromaDB directory, so existing embeddings accumulate (documents are added, not replaced — manually delete `chroma_db/` to reset).

---

### 3.12 `mock_services.py`

**What it does:** Runs a FastAPI server on port 8001 that simulates the real LMS recommendation and assessment APIs. Swap out by changing `RECOMMENDATION_API_URL` and `ASSESSMENT_API_URL` in `config.py` or environment variables.

#### `POST /recommend` — course recommendation logic

```python
difficulty = payload.get("preferred_difficulty") or payload.get("skill_level", "Intermediate")
candidates = [
    c for c in COURSES
    if c["difficulty"] == difficulty
    and c["title"].lower() not in enrolled
]
if len(candidates) < 3:
    candidates = [c for c in COURSES if c["title"].lower() not in enrolled]
candidates.sort(key=lambda c: _score(c, keywords), reverse=True)
return {"courses": [top 3]}
```

Steps:
1. Filter by difficulty (must match exactly: "Beginner", "Intermediate", or "Advanced").
2. Exclude courses the employee is already enrolled in.
3. If fewer than 3 candidates remain after filtering, broaden to all difficulties (all non-enrolled courses).
4. Sort by keyword relevance score.
5. Return top 3.

`_score(course, keywords)` counts how many keywords from the learning_goal + preferred_category appear in the concatenated `title + description + category + skills` string (case-insensitive). This is a simple keyword matching algorithm — not semantic embedding — sufficient for the prototype.

#### `POST /generate` — assessment generation

```python
course_id = payload.get("course_id", "").lower().replace(" ", "-")
questions = ASSESSMENTS.get(course_id)
if not questions:
    questions = [generic reflection questions]
```

The `ASSESSMENTS` dict has entries for 25+ specific courses, keyed by a normalized course ID (lowercase, spaces → dashes). For unknown course IDs, three generic reflection questions are returned.

The `COURSES` catalog includes 25 courses spanning Python, Machine Learning, Deep Learning, AI Agents, and Data Science. All URLs are real, free-to-access courses.

---

### 3.13 `frontend/app.js`

**What it does:** All client-side logic — login, session management, text chat, voice input (VAD), avatar state machine, video playback, and text streaming.

#### State variables

```javascript
let sessionId    = null;   // null until /session/start succeeds
let isProcessing = false;  // true while waiting for any API response
let vadInstance  = null;   // null when mic is off, MicVAD object when active
```

#### Avatar state machine

`setAvatar(state)` applies a CSS class to `avatarContainer` and updates the status label:
- `'idle'` — default; shows the looping silent avatar video.
- `'thinking'` — LLM is working; shows animated "..." bubble.
- `'waiting'` — LLM answered, waiting for video job to complete.
- `'speaking'` — Wav2Lip video is playing.

#### Silero VAD integration

```javascript
vadInstance = await vad.MicVAD.new({
    positiveSpeechThreshold: 0.8,
    negativeSpeechThreshold: 0.5,
    minSpeechFrames:         4,
    preSpeechPadFrames:      2,
    redemptionFrames:        8,
    onSpeechEnd: (audio) => {
        const wavBlob = float32ToWav(audio, 16000);
        sendAudio(wavBlob, 'audio/wav');
    },
});
```

`@ricky0123/vad-web` runs the Silero VAD model (a tiny LSTM classifier) in a Web Worker via ONNX WASM. It processes microphone audio in real time and calls `onSpeechEnd` when it detects that the user has finished speaking — passing a `Float32Array` of the speech segment at 16 kHz.

VAD parameters:
- `positiveSpeechThreshold=0.8`: Silero confidence above this means speech detected. High threshold reduces false positives from background noise.
- `negativeSpeechThreshold=0.5`: Below this means silence. The gap between 0.5 and 0.8 is a hysteresis zone.
- `minSpeechFrames=4`: Minimum speech frames before `onSpeechStart` fires. Prevents very brief pops from triggering.
- `preSpeechPadFrames=2`: Prepend 2 frames before detected speech onset. Prevents clipping the beginning of words.
- `redemptionFrames=8`: After silence is detected, wait 8 more frames before committing to `onSpeechEnd`. Handles natural pauses within speech (e.g., "I want to learn... Python").
- `onVADMisfire`: Called when a speech segment was too short to be real speech. Keeps the mic listening rather than sending.

#### WAV encoding in the browser

```javascript
function float32ToWav(samples, sampleRate) {
    const buf  = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buf);
    // Write RIFF/WAVE header
    str(0, 'RIFF');  view.setUint32(4, 36 + samples.length * 2, true);
    str(8, 'WAVE');  str(12, 'fmt ');
    view.setUint32(16, 16, true);  view.setUint16(20, 1, true);  // PCM
    // ...
    // Convert Float32 samples to Int16
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
}
```

This writes a proper RIFF WAV file header (44 bytes) followed by 16-bit PCM samples. The conversion `s < 0 ? s * 0x8000 : s * 0x7FFF` maps the Float32 range [-1, 1] to Int16 range [-32768, 32767]. This WAV blob is what gets sent to `POST /chat/audio`.

#### Video polling loop

```javascript
async function waitForVideoThenPlay(jobId, replyText) {
    const maxMs = 180000;  // 3-minute timeout
    const start = Date.now();
    while (Date.now() - start < maxMs) {
        await new Promise(r => setTimeout(r, 250));  // poll every 250ms
        const res  = await fetch(`${API}/video/status/${jobId}`);
        const data = await res.json();
        if (data.ready && data.video_url) {
            playVideoAndStreamText(data.video_url, replyText);
            return;
        }
        if (data.error) break;
    }
    // Fallback: show text if video never arrives
    addMessage('assistant', formatMessage(replyText));
    setAvatar('idle');
}
```

Poll interval: 250ms. Maximum wait: 3 minutes (180 seconds). Wav2Lip typically takes 8–20 seconds for a short response. The 3-minute ceiling is for pathological cases (very long responses, GPU contention from multiple users). If the video never arrives, the text reply is displayed as a plain chat bubble — degraded but not broken.

#### Text streaming synchronized to video duration

```javascript
function streamTextIntoBubble(text, bubble, videoDurationSec) {
    const words = streamText.split(' ').filter(Boolean);
    const msPerWord = Math.max(120, (videoDurationSec * 1000) / words.length);
    let i = 0;
    const tick = () => {
        if (i >= words.length) {
            bubble.innerHTML = formatMessage(text);  // replace with clickable links
            return;
        }
        bubble.textContent += (i === 0 ? '' : ' ') + words[i++];
        setTimeout(tick, msPerWord);
    };
    tick();
}
```

`msPerWord = max(120ms, videoDuration / wordCount)` — each word appears at the same pace as the avatar speaks it. Minimum 120ms/word prevents text from appearing faster than any human can speak. When all words are displayed, the bubble's content is replaced with the properly formatted version (including clickable links) via `formatMessage()`.

#### Message formatting

```javascript
function formatMessage(text) {
    const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return escaped
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
        .replace(/\n/g, '<br>');
}
```

Converts markdown links `[Title](url)` to `<a>` tags. HTML-escapes the input first to prevent XSS. The streaming version strips URLs first (bare link labels only) and then replaces with the full formatted version when streaming completes.

---

## 4. The Agent Turn — Full Walkthrough

**Scenario:** Employee types "What is the annual leave policy?" in the chat input.

**Step 1 — Frontend (app.js: `submitText` → `callChat`)**

```javascript
addMessage('user', escapeHtml("What is the annual leave policy?"));
setProcessing(true);
showThinking();  // animated "..." bubble, avatar state → 'thinking'
fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message: "What is the annual leave policy?" })
});
```

**Step 2 — `POST /chat` (web/app.py)**

```python
session = _require_session(request.session_id)  # validates session, updates last_active
agent = session["agent"]
reply = agent.run(request.message)              # synchronous — blocks here
job_id = _start_lipsync_async(reply)            # submits to thread pool, returns immediately
return ChatResponse(session_id=..., reply=reply, video_job_id=job_id)
```

**Step 3 — `HRAgent.run("What is the annual leave policy?")` (brain/agent.py)**

```python
set_profile(self._profile)   # refresh context variable
self.messages.append(("human", "What is the annual leave policy?"))
trimmed = self.messages[-10:]

msgs = [SystemMessage(content=_build_system_prompt(self._profile))]
# SystemMessage contains the 500-char base prompt + the employee's profile block
msgs.append(HumanMessage(content="What is the annual leave policy?"))

response = self.llm_with_tools.invoke(msgs)
# LLM receives: system prompt + user message + tool schemas for 3 tools
# granite4:3b recognizes this as a policy question → calls retrieve_policy
```

The LLM's response has `response.tool_calls = [{"name": "retrieve_policy", "args": {"query": "annual leave policy"}, "id": "..."}]` and `response.content = ""`.

**Step 4 — Path 2: Structured tool call**

```python
tool_call = response.tool_calls[0]
# tool_call = {"name": "retrieve_policy", "args": {"query": "annual leave policy"}}
clean_args = {"query": "annual leave policy"}
raw_result = _TOOLS_MAP["retrieve_policy"].invoke(clean_args)
```

**Step 5 — `retrieve_policy("annual leave policy")` (brain/tools.py)**

```python
docs = rag.retrieve("annual leave policy", k=5)
```

**Step 6 — `RAGManager.retrieve("annual leave policy", k=5)` (brain/rag.py)**

```python
docs = self.vectorstore.similarity_search("annual leave policy", k=5)
```

ChromaDB embeds "annual leave policy" using `FastEmbedEmbeddings` (BAAI/bge-small-en-v1.5, in-process, ~5ms), computes cosine similarity against all stored chunk vectors, returns the 5 most similar `Document` objects from the ingested HR documents.

```python
context = "\n\n".join([doc.page_content for doc in docs])
return f"Relevant policy information:\n{context}"
```

`raw_result` is a multi-paragraph string containing up to 5 x 300-character chunks from the HR policy documents.

**Step 7 — Policy summarization (back in agent.py, Path 2)**

```python
follow_up = msgs + [
    response,          # the AIMessage that contained the tool call
    ToolMessage(content=str(raw_result), tool_call_id=tool_call["id"]),
]
summary = self.llm.invoke(follow_up)  # plain llm, no tools
ai_message = summary.content
```

The LLM receives: system prompt → user question → its own tool call → tool result. It must now synthesize a 1–3 sentence answer grounded in the retrieved context.

Example output: `"You are entitled to 25 days of annual leave per year, which can be taken in blocks or single days with two weeks' notice. Additional leave may be available for long-service employees — contact HR for details."`

**Step 8 — Post-processing (agent.py)**

```python
# Strip any markdown (shouldn't be any here, but safety)
ai_message = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', ai_message)
# ...
# Trim mid-sentence cutoff
ai_message = _trim_to_last_sentence(ai_message)
self.messages.append(("ai", ai_message))

event_logger.log({
    "tool_called": "retrieve_policy",
    "tool_args": {"query": "annual leave policy"},
    "hallucination_guard": False,
    "user_id": "emp_001",
    "input": "What is the annual leave policy?",
    "response": "You are entitled to 25 days...",
    "latency_ms": 2341,
    "grounded": True,
})
return ai_message
```

**Step 9 — Async TTS + Wav2Lip (web/app.py background thread)**

```python
# In _lipsync_job() running in _lipsync_executor
voice.synthesize(reply, output_path="/tmp/voice_abc123.wav")
# XTTS v2 synthesizes speech, saves 24 kHz WAV

lipsync.generate(AVATAR_SILENT_VIDEO, "/tmp/voice_abc123.wav", "/tmp/output_def456.mp4")
# Wav2Lip GAN animates avatar lip sync, saves MP4

_video_jobs[job_id] = {"status": "ready", "video_path": "/tmp/output_def456.mp4"}
os.unlink("/tmp/voice_abc123.wav")  # clean up temp WAV
```

**Step 10 — Frontend receives `/chat` response**

The frontend receives `{reply: "You are entitled to 25 days...", video_job_id: "abc123"}` within ~2–4 seconds.

```javascript
removeThinking();
setAvatar('waiting');
waitForVideoThenPlay("abc123", "You are entitled to 25 days...");
```

Every 250ms, polls `GET /video/status/abc123`. After ~10–15 seconds, gets `{ready: true, video_url: "/video/output_def456.mp4"}`.

**Step 11 — Video playback and text streaming**

```javascript
playVideoAndStreamText("/video/output_def456.mp4", "You are entitled to 25 days...");
// Sets avatar state to 'speaking'
// avatarVideo.src = "http://localhost:8000/video/output_def456.mp4"
// On canplay: avatarVideo.play()
// streamTextIntoBubble(text, bubble, videoDuration)
//   → words appear one by one at videoDuration/wordCount ms per word
// On ended: setAvatar('idle')
```

**Total end-to-end latency breakdown:**
- LLM call (granite4:3b, with tool): ~1.5–3s
- RAG retrieval: ~0.1s
- Summarization LLM call: ~1–2s
- **Text reply visible to user:** ~3–5s from submission
- TTS synthesis (XTTS): ~3–8s
- Wav2Lip generation: ~5–15s
- **Video playing:** ~10–25s from submission

---

## 5. RAG Pipeline

### 5.1 Ingestion Flow

```
HR Documents (hr_docs/ or Azure Blob)
         |
         |  TextLoader / PyPDFLoader / Docx2txtLoader
         v
  List[Document]  (LangChain Document objects)
    page_content: raw text from the file
    metadata: {source: "path/to/file.pdf", page: 3}
         |
         |  RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
         v
  List[Document]  (chunks)
    page_content: 300-char excerpt
    metadata: inherited from parent + chunk index
         |
         |  FastEmbedEmbeddings("BAAI/bge-small-en-v1.5").embed_documents(texts)
         |  (in-process ONNX, ~5ms per chunk)
         v
  List[List[float]]  (384-dim vectors)
         |
         |  ChromaDB.add_documents(chunks)
         |  stores: (id, vector, document_text, metadata)
         |  persists to: ./chroma_db/
         v
  ChromaDB collection  (persisted to disk, survives restarts)
```

**Triggering ingestion:**
```bash
# During development
python -c "from brain.rag import RAGManager; RAGManager().ingest_documents('./hr_docs')"

# Via API (production)
curl -X POST http://localhost:8000/admin/ingest \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"local_path": "./hr_docs"}'
```

### 5.2 Retrieval Flow

```
User query: "annual leave policy"
         |
         |  FastEmbedEmbeddings("BAAI/bge-small-en-v1.5").embed_query(query)
         |  ~5ms
         v
  query_vector: [0.123, -0.456, ..., 0.789]  (384 dimensions)
         |
         |  ChromaDB.similarity_search(query, k=5)
         |  cosine similarity vs. all stored chunk vectors
         v
  Top-5 Document objects (closest cosine similarity)
    doc[0].page_content: "Annual leave entitlement is 25 days per calendar year..."
    doc[1].page_content: "Leave requests must be submitted at least two weeks..."
    ...
         |
         |  "\n\n".join([doc.page_content for doc in docs])
         v
  Combined context string (up to 5 x 300 chars = 1500 chars max)
         |
         |  return f"Relevant policy information:\n{context}"
         v
  Tool result -> LLM summarization -> final answer
```

---

## 6. Tool Execution Logic — All Three Paths

### Path 1: Leaked tool call

**When it happens:** The model outputs a tool call as plain text in `response.content` instead of the structured `response.tool_calls` list. Observed with granite4, llama3.1, and older mistral versions.

**Two sub-forms:**

**A. Bare tool name:**
```python
if stripped in _TOOLS_MAP:  # e.g. stripped == "retrieve_policy"
    if tool_name == "retrieve_policy":
        args = {"query": user_input}     # use original user input as query
    elif tool_name == "recommend_courses":
        args = {"learning_goal": user_input}
    else:
        args = {}
    return _TOOLS_MAP[tool_name].invoke(args)
```
The bare name gives us no arguments. We reconstruct plausible args from the original user input. This is a best-effort fallback.

**B. Raw JSON:**
```python
data = json.loads(stripped)
tool_name = data.get("name")
args = data.get("parameters") or data.get("arguments") or {}
# "parameters" is llama3.1's key; "arguments" is mistral/OpenAI's key
args = {k: v for k, v in args.items() if v is not None}
return _TOOLS_MAP[tool_name].invoke(args)
```
Handles both `"parameters"` and `"arguments"` key names because different models use different conventions. Strips `None` values before invoking.

**Result:** The tool output is processed by `_phrase_tool_result()` (markdown stripping only, no LLM). `_needs_sentence_trim = False` because the tool output is a complete structured list.

### Path 2: Structured tool call (normal)

**When it happens:** The model correctly uses the tool-calling protocol, populating `response.tool_calls`.

```python
tool_call = response.tool_calls[0]  # take only the FIRST
```

Why only the first? `granite4` occasionally attempts parallel tool calls (e.g., calling both `retrieve_policy` and `recommend_courses` for an ambiguous query). Executing all of them would produce multiple results that the LLM would need to reconcile, adding a third LLM call and significant latency. Taking only the first is a deliberate simplification — in practice, the system prompt is specific enough that the model should never need multiple tools for a single query.

**For `retrieve_policy`:** two LLM calls total (one to decide the tool call, one to summarize the result). The second call uses `self.llm` (no tools), preventing further tool invocations.

**For `recommend_courses` and `generate_assessment`:** one LLM call total. The tool result is formatted directly via `_phrase_tool_result()`.

### Path 3: Direct answer (no tool call)

**When it happens:** The model decides to answer without calling a tool. Legitimate cases: greetings, clarification requests, off-topic questions. Illegitimate case: hallucinated course/policy answers.

**Hallucination guard:**
```python
_course_openers = ("here are some", "here's some", "recommended course",
                   "i recommend", "you might want to")
if any(p in ai_message.lower() for p in _course_openers):
    result = _TOOLS_MAP["recommend_courses"].invoke({"learning_goal": user_input})
    ai_message = _phrase_tool_result(self.llm, result)
```

If the direct answer looks like a course recommendation, it was invented from the model's training data — not from the LMS catalog. The guard overrides the answer and forces the correct tool call. This is a heuristic: it catches the most common hallucination pattern (course lists) but not policy hallucinations, which are harder to detect without domain knowledge.

**Empty/fallback guard:**
```python
_fallback_phrases = ("i couldn't find", "please contact hr", "no courses found")
if not ai_message.strip() or any(p in ai_message.lower() for p in _fallback_phrases):
    ai_message = "Sorry, I didn't quite catch that — could you rephrase your question?"
```
If the model emits an empty string or one of the standard failure phrases without having called a tool first, replace with a generic clarification request. This prevents displaying "I couldn't find" to the user when the model failed to call the RAG tool at all.

---

## 7. Session Lifecycle

```
LMS Backend                    FastAPI (/session/start)           brain/session.py
     |                                  |                               |
     |-- POST /session/start ---------->|                               |
     |   Authorization: Bearer secret   |-- create_session(profile) -->|
     |   Body: UserProfile JSON         |                               |-- HRAgent() created
     |                                  |                               |-- session stored in _store
     |<- {session_id: "sess_abc..."} ---|<- session_id ----------------|
     |                                  |                               |
     |                                  |-- get_session(id) ---------->|
     |                                  |-- agent.set_profile(profile)  |
     |                                  |                               |

LMS Frontend                   FastAPI (/session/welcome, /chat)  brain/session.py
     |                                  |                               |
     |-- POST /session/welcome -------->|-- get_session(id) ---------->|
     |<- {reply, video_url} ------------|    updates last_active        |
     |                                  |                               |
     |-- POST /chat (per turn) -------->|-- get_session(id) ---------->|
     |                                  |    updates last_active        |
     |<- {reply, video_job_id} ---------|                               |
     |                                  |                               |
     |  (after 60min inactivity)        |-- get_session(id) ---------->|
     |                                  |    last_active < cutoff       |
     |<- HTTP 404 ----------------------|    delete_session(id)         |
     |                                  |    return None                |
     |                                  |                               |
     |-- DELETE /session/{id} -------->|-- delete_session(id) ------->|
     |<- {message: "Session ended."} --|    _store.pop(id)             |
```

**Key properties:**
- Each session has its own `HRAgent` instance with its own `self.messages` list. Conversation history is completely isolated between sessions/users.
- The TTL check happens lazily on every `get_session()` call — there is no background cleanup task. Expired sessions remain in `_store` until someone calls `get_session()` for them. In practice this is fine because the LMS will always call `DELETE /session/{id}` on logout.
- There is no maximum session count. Under heavy load, many concurrent `HRAgent` instances means many LangChain/Ollama connection objects in memory. This is a known limitation for the current single-server architecture.

---

## 8. GPU Memory Management

Understanding this section is essential for debugging OOM errors and for extending the system to support additional models.

### Memory Budget (typical values, may vary by GPU)

| Component | VRAM Usage | When loaded | When unloaded |
|---|---|---|---|
| Whisper large-v3 | ~3.8 GB | Loaded in `__init__`, unloaded after each `transcribe()` | Freed via `del self.model; torch.cuda.empty_cache()` |
| XTTS v2 | ~2.0 GB | Loaded at startup, stays resident | Never unloaded |
| Wav2Lip GAN | ~0.5 GB | Loaded at startup, stays resident | Never unloaded |
| RetinaFace | ~0.1 GB | Loaded at startup, stays resident | Never unloaded |
| granite4:3b (Ollama) | ~2.5 GB | Loaded per inference, `keep_alive=0` unloads immediately | Freed by Ollama after each response |

**At startup (worst case):** Whisper (3.8) + XTTS (2.0) + Wav2Lip (0.6) + granite4 (2.5 during first inference) = 8.9 GB. This requires a GPU with at least 10 GB VRAM to avoid OOM at startup. On an 8 GB GPU, Whisper must be unloaded before the first LLM call.

**During steady-state operation (per turn):**
1. `POST /chat` arrives. LLM call: granite4 loads (~2.5 GB). XTTS + Wav2Lip still resident. Total: ~5.1 GB. Granite4 unloads after response (`keep_alive=0`).
2. TTS + lipsync job starts: XTTS synthesizes (~2.0 GB), `torch.cuda.empty_cache()` after. Wav2Lip renders (~0.6 GB). Total peak: ~2.6 GB. No Whisper, no Ollama.

**During audio turn:**
1. `POST /chat/audio` arrives. Whisper loads (~3.8 GB). XTTS + Wav2Lip resident. Total: ~6.4 GB. Whisper unloads immediately after transcription.
2. LLM call: granite4 loads (~2.5 GB). No Whisper. Total: ~5.1 GB. Granite4 unloads.
3. TTS + lipsync: same as above.

**Why `keep_alive=0` for Ollama:** Without this, Ollama keeps granite4 loaded for its default TTL (5 minutes). During that window, the LLM + XTTS + Wav2Lip would all be resident simultaneously (~5.1 GB), pushing the system over the limit on a 6–8 GB GPU.

**Why Whisper is loaded/unloaded but XTTS is not:** XTTS is called on every response (every turn), so the overhead of loading it each time (5–10 seconds) would be prohibitive. Whisper is only needed for audio turns, which are less frequent, and the reload cost (~3–5 seconds from HuggingFace cache) is acceptable. Keeping both resident simultaneously would require ~5.8 GB just for AI inference models before counting Ollama.

**What breaks if you change the GPU management strategy:**
- **Remove `keep_alive=0`:** Ollama holds granite4 during TTS+Wav2Lip, likely causing OOM on 8 GB GPU.
- **Keep Whisper loaded permanently:** Audio + TTS + Wav2Lip simultaneous = ~6.5 GB, workable on 8 GB but leaves no headroom. First LLM call after audio will OOM.
- **Add a second concurrent synthesis worker:** Two XTTS runs = ~4 GB. Combined with Wav2Lip = ~4.6 GB. Without Ollama loaded this is fine, but with it active it OOMs.
- **Increase `num_predict` significantly:** Longer responses → longer audio → Wav2Lip processes more frames → peak VRAM during lipsync may exceed GPU capacity.

---

## 9. Frontend ↔ Backend Protocol

### 9.1 All API Calls

| Call | Method | Path | Auth | Body | Response |
|---|---|---|---|---|---|
| Create session | POST | `/session/start` | `Authorization: Bearer <secret>` | `UserProfile` JSON | `{session_id, message}` |
| Welcome video | POST | `/session/welcome` | None | `{session_id}` | `ChatResponse` (video_url set) |
| Text chat | POST | `/chat` | None | `{session_id, message}` | `ChatResponse` (video_job_id set) |
| Audio chat | POST | `/chat/audio` | None | FormData: `session_id`, `audio` file | `ChatResponse` (video_job_id set) |
| Video status | GET | `/video/status/{job_id}` | None | — | `VideoJobStatus` |
| Video stream | GET | `/video/{video_id}` | None | Range header | MP4 stream (206) |
| End session | DELETE | `/session/{session_id}` | None | — | `{message}` |
| Health | GET | `/health` | None | — | `{status, active_sessions}` |
| Admin ingest | POST | `/admin/ingest` | `Authorization: Bearer <secret>` | `IngestRequest` JSON | `{chunks_ingested, status}` |

### 9.2 `ChatResponse` schema

```json
{
  "session_id": "sess_abc123def456",
  "reply": "You are entitled to 25 days of annual leave per year.",
  "transcription": null,
  "video_url": null,
  "video_job_id": "f3a9b2c1..."
}
```

`transcription` is populated for `/chat/audio` turns only — shown in the chat UI so the employee can verify what was heard. `video_url` and `video_job_id` are mutually exclusive: `video_url` for the synchronous welcome video, `video_job_id` for all chat turns.

### 9.3 Async video polling sequence

```
Browser                          FastAPI
  |-- POST /chat -------------->|  (2-5s LLM call)
  |<- {reply, video_job_id} ----|  (returns immediately)
  |                              |
  |  [show thinking bubble]      |  [background: TTS + Wav2Lip in executor]
  |                              |
  |-- GET /video/status/abc --->|
  |<- {ready: false} -----------|  (job still pending/running)
  |  [wait 250ms]                |
  |-- GET /video/status/abc --->|
  |<- {ready: false} -----------|
  |  ... (repeat ~40-80 times)  |
  |-- GET /video/status/abc --->|
  |<- {ready: true,             |  (job complete)
  |    video_url: "/video/..."} |
  |                              |
  |-- GET /video/output_xyz.mp4 >|  (browser-initiated Range request)
  |<- HTTP 206 Partial Content --|
  |  [video plays + text streams]|
```

### 9.4 `UserProfile` fields

```json
{
  "user_id":          "emp_001",
  "name":             "Abiola K.",
  "job_role":         "Data Analyst",
  "department":       "Engineering",
  "skill_level":      "Intermediate",
  "learning_goal":    null,
  "preferred_category": null,
  "preferred_difficulty": "Beginner",
  "preferred_duration":   "Short",
  "known_skills":     ["SQL", "Python"],
  "enrolled_courses": [],
  "context":          "avatar_chat"
}
```

`skill_level` must be exactly `"Beginner"`, `"Intermediate"`, or `"Advanced"` — these values are passed to the recommendation API which filters by exact match.

---

## 10. Data Flow Diagrams — The Three Tool Types

### 10.1 Policy Retrieval (`retrieve_policy`)

```
User: "What is the annual leave policy?"
         |
         v
  HRAgent.run()
  llm_with_tools.invoke(msgs)
         |
         v  tool_calls[0].name == "retrieve_policy"
  retrieve_policy(query="annual leave policy")
         |
         v
  RAGManager.retrieve("annual leave policy", k=5)
  FastEmbed: embed query -> 384-dim vector
  ChromaDB: cosine similarity search -> top-5 Document chunks
         |
         v
  "Relevant policy information:\n
   Annual leave entitlement is 25 days...\n\n
   Leave requests must be submitted...\n\n
   ..."
         |
         v  (raw_result — multi-paragraph, too long for TTS)
  follow_up = msgs + [AIMessage(tool_call), ToolMessage(raw_result)]
  self.llm.invoke(follow_up)    <- second LLM call, no tools
         |
         v
  "You are entitled to 25 days of annual leave per year, which can
   be taken in blocks or single days with two weeks' notice."
         |
         v
  _trim_to_last_sentence()     <- trim if num_predict cut it off
  event_logger.log({grounded: true, tool_called: "retrieve_policy"})
  return answer
```

### 10.2 Course Recommendation (`recommend_courses`)

```
User: "I want to learn machine learning, something intermediate level"
         |
         v
  HRAgent.run()
  llm_with_tools.invoke(msgs)
         |
         v  tool_calls[0].name == "recommend_courses"
  recommend_courses(
      learning_goal="machine learning",
      preferred_difficulty="Intermediate",
      preferred_duration=None,
      preferred_category="machine learning"
  )
         |
         v
  get_profile() -> {user_id: "emp_001", job_role: "Data Analyst", ...}
  payload = {user_id, name, job_role, ..., learning_goal, preferred_difficulty, ...}
  POST http://localhost:8001/recommend (timeout=10s)
         |
         v
  mock_services /recommend:
    filter COURSES by difficulty="Intermediate", exclude enrolled
    score by keywords ["machine", "learning"]
    return top-3: [
      "Machine Learning Specialization -- Andrew Ng",
      "fast.ai: Practical Machine Learning for Coders",
      "Kaggle: Intermediate Machine Learning"
    ]
         |
         v
  result = "Here are some recommended courses:\n
    [Machine Learning Specialization...](url): description\n
    ..."
         |
         v  (structured list -- no second LLM call needed)
  _phrase_tool_result(): strip markdown symbols
  -> "Machine Learning Specialization -- Andrew Ng: The gold-standard ML course..."
  _needs_sentence_trim = False
  return answer
```

### 10.3 Assessment Generation (`generate_assessment`)

```
User: "I finished the Kaggle Python course, can you test me?"
         |
         v
  HRAgent.run()
  LLM: no course_id provided -- asks for clarification
  -> "Sure! Could you give me the course ID or name for the Kaggle Python course?"
         |
  User: "kaggle-python"
         |
         v
  HRAgent.run()
  llm_with_tools.invoke(msgs)  <- conversation history includes previous turn
         |
         v  tool_calls[0].name == "generate_assessment"
  generate_assessment(course_id="kaggle-python")
         |
         v
  POST http://localhost:8001/generate  {course_id: "kaggle-python"}
         |
         v
  mock_services /generate:
    ASSESSMENTS["kaggle-python"] = [
      {question: "What does a list comprehension like [x*2 ...]?",
       options: ["[0, 2, 4]", "[1, 2, 3]", ...]},
      ...3 questions
    ]
         |
         v
  result = "Here is your assessment:\n1. Question?\n   - opt1\n   - opt2\n..."
         |
         v  (structured list -- no second LLM call)
  _phrase_tool_result(): strip numbers and bullet dashes
  -> "Question?\nopt1\nopt2\n..."
  _needs_sentence_trim = False
  return answer
```

---

## 11. Design Decisions and Trade-offs

### Single-pass agent over ReAct loop

**Decision:** `HRAgent.run()` makes exactly one LLM call (two for `retrieve_policy`) and returns. There is no loop where the LLM can call multiple tools in sequence.

**Why:** ReAct (Reasoning + Acting) loops work well when each tool call reveals information needed to decide the next action. For this HR assistant, every query maps to exactly one tool. When the system used a ReAct loop, granite4 entered loops of 4–6 sequential tool calls (calling `retrieve_policy` three times with slightly different queries), adding 57 seconds of latency. The single-pass approach eliminates this entirely. The trade-off: genuinely multi-step queries (e.g., "What's the leave policy AND recommend me some HR management courses") would require two turns. This is an acceptable limitation for a chat UI.

### FastEmbed over OllamaEmbeddings

**Decision:** `FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")` instead of `OllamaEmbeddings(model="nomic-embed-text")`.

**Why:** `OllamaEmbeddings` makes an HTTP POST to `localhost:11434/api/embeddings` for every document chunk during ingestion and every query during retrieval. At ~4.6s per chunk, ingesting 100 chunks takes 7.7 minutes. `FastEmbedEmbeddings` runs ONNX Runtime in-process — no HTTP, no Ollama dependency — embedding 100 chunks takes ~0.5 seconds. Query latency drops from ~500ms to ~5ms. The trade-off: FastEmbed uses a different model (`bge-small-en-v1.5` vs `nomic-embed-text`). This is fine because embeddings only need to be consistent within the system — ingestion and retrieval use the same model instance.

### `keep_alive=0` for Ollama

**Decision:** `ChatOllama(keep_alive=0)` unloads the model immediately after each response.

**Why:** VRAM budget. granite4:3b uses ~2.5 GB. XTTS uses ~2.0 GB. On an 8 GB GPU, they cannot coexist. Setting `keep_alive=0` means Ollama frees the VRAM the instant the token stream ends. XTTS can then run without OOM. The trade-off: the next LLM call incurs a 2–4 second reload penalty (from Ollama's model cache on disk — already loaded when the first call warmed it). This is why the text reply latency includes that overhead. Using a smaller model (e.g., granite4:1b) would reduce VRAM and potentially allow `keep_alive` to be nonzero.

### `num_predict=200`

**Decision:** Hard cap on LLM output at 200 tokens.

**Why:** Four reasons compound each other:
1. Lipsync quality degrades with long audio — Wav2Lip temporal consistency breaks down over 30+ second clips.
2. XTTS VRAM usage scales with audio length — very long responses can OOM during synthesis.
3. TTS synthesis time scales linearly — a 500-token response takes 3–5x longer to synthesize than a 200-token one, making the total wait unbearable.
4. Conversational HR responses should be 1–3 sentences anyway — 200 tokens is generous.

The trade-off: complex policy information may be truncated. The `_trim_to_last_sentence()` function catches mid-sentence cutoffs, and the system prompt instructs the model to be concise.

### `chunk_size=300, chunk_overlap=100`

**Decision:** 300-character chunks with 100-character overlap for the RAG text splitter.

**Why:** The LLM context window is 2048 tokens. Five retrieved chunks (at 300 chars each, approximately 75 tokens each) use 375 tokens, leaving plenty of room for the system prompt (~100 tokens), conversation history, and the summarization instruction. Shorter chunks (e.g., 100 chars) would require more chunks to cover the same policy text. Longer chunks (e.g., 1000 chars) would push the context limit and might mix unrelated policy content within one chunk. The 100-char overlap prevents a key sentence from being stranded exactly at a chunk boundary.

### Pre-computed XTTS speaker latents

**Decision:** Compute `gpt_cond_latent` and `speaker_embedding` once at startup and cache them.

**Why:** Each call to `tts.tts(text, speaker_wav=...)` reloads and reprocesses the reference WAV file — a 2–3 second overhead. With ~10 turns in a typical session, that's 20–30 seconds of avoidable work. The latents are deterministic given the same WAV file, so computing them once at startup and reusing them is exactly equivalent. The trade-off: the speaker WAV cannot be hot-swapped mid-session. Changing `hr_voice_sample.wav` requires a server restart.

### Browser-side VAD (Silero WASM)

**Decision:** Voice activity detection runs in the browser, not on the server.

**Why:** Server-side VAD would require streaming audio chunks to the backend continuously, with the server deciding when speech ends and accumulating the audio. This doubles the audio latency (upload time + VAD time on server) and wastes bandwidth on silence. Browser-side Silero VAD runs at ~1ms per 30ms audio frame on modern CPUs, produces zero network traffic during silence, and delivers only the speech segment as a single WAV upload when the user finishes speaking. The trade-off: the WASM model (~1.8 MB) adds a one-time download cost and requires a browser that supports WASM (all modern browsers do).

### Silent video as Wav2Lip face input

**Decision:** Use `hr_avatar_silent.mp4` (looping video) rather than `hr_avatar.jpg` (still image) as the face input to Wav2Lip.

**Why:** Wav2Lip's temporal consistency mechanism works by blending face crops across frames. When the input is a still image repeated N times, all frames are identical, and the only variation is the lip movement — which can look mechanical. A short looping video with natural head movement (nodding, slight tilts) provides temporal diversity that Wav2Lip can work with, producing more natural-looking animation. The trade-off: the face video loop may not align perfectly with the audio (a blink might happen at an awkward moment). This is acceptable for a prototype.

### In-process Wav2Lip (no subprocess)

**Decision:** `face.py` imports `inference.py` and calls `_inf.main()` directly, rather than spawning `python inference.py` as a subprocess.

**Why:** The original Wav2Lip script loads two large models (Wav2Lip GAN and RetinaFace) from disk on every invocation. Via subprocess, this overhead is 5–7 seconds per video. In-process loading happens once at startup; subsequent calls only run the inference computation. The trade-off: the integration is brittle — it depends on implementation details of `inference.py` (module-level globals, `args` namespace, relative `temp/` directory). If Wav2Lip is updated, the integration may break and require updating `do_load()` and the `args` namespace injection.

### In-memory sessions (no database)

**Decision:** `_store` in `brain/session.py` is a plain Python dictionary in RAM.

**Why:** Simplicity. Adding Redis or PostgreSQL would require additional infrastructure. For a single-server deployment supporting tens of concurrent users, RAM is sufficient and lookup is O(1). The trade-off: sessions are lost on server restart, and the store is not shared across multiple server instances. In production, Redis would be the natural replacement.

---

## 12. Reproduction Checklist

Follow these steps in order to go from zero to a working system.

### Prerequisites

- Linux (tested on Ubuntu 22.04+)
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ and at least 8 GB VRAM (see GPU section for caveats)
- `ffmpeg` installed and on PATH: `sudo apt install ffmpeg`
- `ollama` installed and running: follow https://ollama.com/download

### Step 1 — Clone and create virtualenv

```bash
git clone <repo-url> hr_avatar
cd hr_avatar
python3 -m venv hr_venv
source hr_venv/bin/activate
pip install --upgrade pip
```

### Step 2 — Install Python dependencies

```bash
pip install fastapi uvicorn[standard] langchain langchain-community langchain-ollama
pip install chromadb fastembed
pip install faster-whisper
pip install TTS  # Coqui XTTS v2
pip install azure-storage-blob  # optional -- only for Azure ingestion
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install soundfile numpy requests python-multipart
```

Install Wav2Lip Python dependencies:
```bash
cd face/wav2lip
pip install -r requirements.txt
cd ../..
```

### Step 3 — Pull the Ollama model

```bash
ollama pull granite4:3b
# Verify:
ollama list
```

### Step 4 — Download the Wav2Lip checkpoint

```bash
curl -L 'https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true' \
  -o wav2lip_gan.pth
```

This file must be at `hr_avatar/wav2lip_gan.pth` (the project root, not inside `face/wav2lip/`).

### Step 5 — Prepare avatar assets

Place the following files in `assets/`:
- `hr_avatar.jpg` — a portrait photo of the HR avatar (face must be clearly visible, no sunglasses, well lit)
- `hr_avatar_silent.mp4` — a short (5–10 second) looping video of the avatar with natural head movement, no speech
- `hr_voice_sample.wav` — 6–30 seconds of clean speech in the avatar's voice (no background noise, 16–48 kHz mono or stereo)

```bash
ls assets/
# Should show: hr_avatar.jpg  hr_avatar_silent.mp4  hr_voice_sample.wav
```

### Step 6 — Ingest HR documents

Place HR policy documents in `hr_docs/`:
```bash
mkdir hr_docs
# Copy your .txt, .pdf, or .docx HR documents into hr_docs/
```

Ingest before or after starting the server:
```bash
python -c "
from brain.rag import RAGManager
rag = RAGManager()
rag.ingest_documents('./hr_docs')
print('Done')
"
```

Or via the API after starting the server (Step 8):
```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"local_path": "./hr_docs"}'
```

### Step 7 — Start the mock LMS services (development only)

In a separate terminal:
```bash
source hr_venv/bin/activate
python mock_services.py
# Starts on http://localhost:8001
```

### Step 8 — Start the main server

```bash
source hr_venv/bin/activate
python web/app.py
# Starts on http://localhost:8000
# First startup: pre-renders welcome.mp4 (~30-60 seconds)
# Subsequent startups: skips pre-render if welcome.mp4 exists
```

Watch the logs for:
```
Loading Whisper 'large-v3' on cuda...
Whisper model loaded
Loading XTTS on cuda...
Pre-computing speaker conditioning latents...
XTTS loaded with cached speaker latents
Wav2Lip + RetinaFace models loaded and cached in memory
Pre-rendering welcome video...   (first startup only)
Welcome video pre-rendered and cached.
```

If you see CUDA OOM errors during startup, see Section 8.

### Step 9 — Open the frontend

Open `frontend/index.html` directly in a browser, or serve it:
```bash
cd frontend
python -m http.server 3000
# Then open http://localhost:3000/index.html
```

The frontend has `const API = 'http://localhost:8000'` hardcoded — ensure the backend is running on port 8000.

### Step 10 — Test the full pipeline

1. Fill in the login form (defaults are pre-filled with a sample employee profile).
2. Click "Start Conversation". The welcome video should play within ~1–2 seconds (served from cache).
3. Type "What is the annual leave policy?" and press Enter.
   - Text reply appears in ~3–5 seconds.
   - Avatar video plays ~10–20 seconds after submission.
4. Click the microphone button, speak a question, stop speaking — it should auto-send.
5. Click "End Session" to verify session cleanup.

### Step 11 — Verify event logging

```bash
tail -f logs/events.jsonl | python -m json.tool
# Should show one JSON record per conversation turn
```

### Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` at startup | GPU has less than 8 GB VRAM or other processes are using GPU | Free GPU memory: `nvidia-smi`, kill other processes. Consider `model_size="medium"` for Whisper. |
| `FileNotFoundError: wav2lip_gan.pth` | Checkpoint not downloaded | Run Step 4 |
| `No relevant policy documents found` | ChromaDB is empty | Run ingestion (Step 6) |
| `HTTP 401` on session start | Wrong or missing Bearer token | Check `LMS_SHARED_SECRET` env var matches `const SECRET` in `app.js` |
| Video never plays (stuck on "Preparing...") | Wav2Lip job failed | Check `logs/hr_avatar.log` for the error in `_lipsync_job` |
| TTS sounds wrong or distorted | Speaker WAV is too short or has background noise | Replace `assets/hr_voice_sample.wav` with 15–30 seconds of clean speech |
| Transcription is garbage | Whisper could not understand the audio | Check microphone input, ensure VAD is sending audible speech |
| `Connection refused` on `/recommend` | `mock_services.py` not running | Start Step 7 in a separate terminal |
| Model outputs tool call in content | Model-specific leaked tool call quirk | Already handled by `_try_execute_leaked_tool_call`. For a new model with a novel format, extend the parser in `agent.py`. |

### Environment variables for production

```bash
export LMS_SHARED_SECRET="your-secure-random-secret"
export RECOMMENDATION_API_URL="https://your-lms.example.com/api/recommend"
export ASSESSMENT_API_URL="https://your-lms.example.com/api/assess"
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
export AZURE_STORAGE_CONTAINER="hr-documents"
export LOG_LEVEL="INFO"
```

Start with:
```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 1
# workers=1 is mandatory -- the GPU executor is not safe for multiple worker processes
```
