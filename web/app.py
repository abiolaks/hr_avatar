# web/app.py
# FastAPI server — the bridge between the LMS and the HR Avatar pipeline.
#
# Flow:
#   1. LMS backend calls POST /session/start with the employee's profile.
#      Returns a session_id the LMS frontend stores.
#   2. On each conversation turn the LMS frontend calls POST /chat (text)
#      or POST /chat/audio (microphone recording).
#   3. Avatar returns the text reply + a video URL for the lip-synced response.
#   4. LMS frontend embeds the video and plays it for the employee.

import asyncio
import os
import sys
import uuid
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from config import AVATAR_SILENT_VIDEO, ASSETS_DIR
from brain.rag import RAGManager
from brain.session import create_session, delete_session, get_session, active_session_count
from transcriber.transcriber import Transcriber
from voice.voice import VoiceSynthesizer
from face.face import LipSyncGenerator
from logger import logger

# ── Shared heavy modules (loaded once at startup) ─────────────────────────────
transcriber = Transcriber()
voice = VoiceSynthesizer()
lipsync = LipSyncGenerator()

# ── Welcome video (pre-rendered once at startup, served as a static file) ────
_WELCOME_GREETING = (
    "Welcome! I'm your HR Avatar — I'm here to help you with company policies, "
    "course recommendations, and knowledge assessments. What can I help you with today?"
)
_WELCOME_VIDEO_PATH = os.path.join(ASSETS_DIR, "welcome.mp4")


def _render_welcome_video() -> None:
    temp_voice = f"/tmp/welcome_voice_{uuid.uuid4().hex}.wav"
    try:
        voice.synthesize(_WELCOME_GREETING, output_path=temp_voice)
        lipsync.generate(AVATAR_SILENT_VIDEO, temp_voice, _WELCOME_VIDEO_PATH)
        logger.info("Welcome video pre-rendered and cached.")
    finally:
        if os.path.exists(temp_voice):
            os.unlink(temp_voice)


# Single-worker executor serialises all TTS+Wav2Lip jobs so they don't race on the GPU.
# max_workers=1 is intentional — these models share VRAM and cannot run in parallel.
_lipsync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# job_id → {"status": "pending"|"ready"|"error", "video_path": str, "error": str}
_video_jobs: dict = {}

app = FastAPI(
    title="HR Avatar API",
    description="LMS-integrated conversational HR assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.on_event("startup")
async def startup_tasks():
    if not os.path.exists(_WELCOME_VIDEO_PATH):
        logger.info("Pre-rendering welcome video...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _render_welcome_video)
    else:
        logger.info("Cached welcome video found — skipping render.")


# Shared secret the LMS must send to create sessions.
# Set via env var LMS_SHARED_SECRET; defaults to "dev-secret" for local dev.
_SHARED_SECRET = os.getenv("LMS_SHARED_SECRET", "dev-secret")


# ── Pydantic models ────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    user_id: str
    name: str
    job_role: str
    department: str
    skill_level: str = "Beginner"
    learning_goal: Optional[str] = None
    preferred_category: Optional[str] = None
    preferred_difficulty: Optional[str] = "Beginner"
    preferred_duration: Optional[str] = "Short"
    known_skills: List[str] = Field(default_factory=list)
    enrolled_courses: List[str] = Field(default_factory=list)
    context: str = "avatar_chat"


class SessionStartResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    transcription: Optional[str] = None  # populated for audio turns — shown in chat UI
    video_url: Optional[str] = None      # set for synchronous video (welcome)
    video_job_id: Optional[str] = None   # set for async video — poll /video/status/{id}


class VideoJobStatus(BaseModel):
    ready: bool
    video_url: Optional[str] = None
    error: Optional[str] = None


# ── Helper ─────────────────────────────────────────────────────────────────────

def _require_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return session


def _lipsync_job(job_id: str, reply: str, temp_voice: str, video_path: str) -> None:
    """Background worker: TTS → Wav2Lip.  Runs in _lipsync_executor (max 1 worker)."""
    try:
        voice.synthesize(reply, output_path=temp_voice)
        lipsync.generate(AVATAR_SILENT_VIDEO, temp_voice, video_path)
        _video_jobs[job_id] = {"status": "ready", "video_path": video_path}
        logger.info(f"Lipsync job {job_id} done: {video_path}")
    except Exception as exc:
        _video_jobs[job_id] = {"status": "error", "error": str(exc)}
        logger.error(f"Lipsync job {job_id} failed: {exc}", exc_info=True)
    finally:
        if os.path.exists(temp_voice):
            os.unlink(temp_voice)


def _start_lipsync_async(reply: str) -> str:
    """
    Submits TTS+Wav2Lip to the background executor and returns the job_id immediately.
    The caller returns the LLM text to the client without waiting for video generation.
    """
    job_id     = uuid.uuid4().hex
    video_path = f"/tmp/output_{uuid.uuid4().hex}.mp4"
    temp_voice = f"/tmp/voice_{uuid.uuid4().hex}.wav"
    _video_jobs[job_id] = {"status": "pending"}
    _lipsync_executor.submit(_lipsync_job, job_id, reply, temp_voice, video_path)
    return job_id


def _run_lipsync_sync(reply: str) -> str:
    """Synchronous TTS+Wav2Lip — used only for the welcome greeting."""
    video_path = f"/tmp/output_{uuid.uuid4().hex}.mp4"
    temp_voice = f"/tmp/voice_{uuid.uuid4().hex}.wav"
    try:
        voice.synthesize(reply, output_path=temp_voice)
        lipsync.generate(AVATAR_SILENT_VIDEO, temp_voice, video_path)
    finally:
        if os.path.exists(temp_voice):
            os.unlink(temp_voice)
    return video_path


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": active_session_count()}


@app.post("/session/start", response_model=SessionStartResponse)
def session_start(
    profile: UserProfile,
    authorization: str = Header(..., description="Bearer <LMS_SHARED_SECRET>"),
):
    """
    Called by the LMS backend when an employee opens the Avatar widget.
    Accepts the full employee profile so the Avatar never needs to ask
    for job role, department, or enrolled courses.

    Returns a session_id that the LMS frontend must include in all
    subsequent /chat requests.
    """
    # Verify the shared secret
    if not authorization.startswith("Bearer ") or authorization[7:] != _SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing shared secret.")

    session_id = create_session(profile.model_dump())

    # Inject profile into the agent so system prompt is personalized
    session = get_session(session_id)
    session["agent"].set_profile(profile.model_dump())

    logger.info(f"/session/start | session: {session_id} | user: {profile.user_id}")
    return SessionStartResponse(
        session_id=session_id,
        message=f"Session started for {profile.name}.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Text-based conversation turn.
    Returns the LLM reply immediately; TTS+Wav2Lip run in the background.
    Poll GET /video/status/{video_job_id} for the lip-sync video URL.
    """
    session = _require_session(request.session_id)
    agent = session["agent"]

    logger.info(f"/chat | session: {request.session_id} | msg: {request.message[:80]}")

    reply    = agent.run(request.message)          # LLM only — fast path
    job_id   = _start_lipsync_async(reply)         # TTS+Wav2Lip in background

    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        video_job_id=job_id,
    )


@app.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(..., description="Audio recording from the employee's microphone"),
):
    """
    Audio-based conversation turn.
    session_id must be sent as a form field alongside the audio file.
    Transcribes audio then runs the full pipeline synchronously.
    """
    session = _require_session(session_id)
    agent = session["agent"]

    temp_audio = f"/tmp/upload_{uuid.uuid4().hex}{os.path.splitext(audio.filename or '.webm')[1]}"
    with open(temp_audio, "wb") as f:
        f.write(await audio.read())

    try:
        text = transcriber.transcribe(temp_audio)
        logger.info(f"/chat/audio | session: {session_id} | transcribed: {text[:80]!r}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio transcription failed: {e}")
    finally:
        if os.path.exists(temp_audio):
            os.unlink(temp_audio)

    if not text.strip():
        # VAD sent a clip with no recognisable speech (background noise, short clip, etc.)
        # Return a polite prompt without invoking the LLM.
        nudge = "Sorry, I didn't catch that. Could you say that again?"
        job_id = _start_lipsync_async(nudge)
        return ChatResponse(session_id=session_id, reply=nudge, video_job_id=job_id)

    reply  = agent.run(text)           # LLM only — fast path
    job_id = _start_lipsync_async(reply)   # TTS+Wav2Lip in background

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        transcription=text,
        video_job_id=job_id,
    )


@app.get("/video/status/{job_id}", response_model=VideoJobStatus)
def video_status(job_id: str):
    """
    Poll this after receiving a video_job_id from /chat or /chat/audio.
    Returns {ready: true, video_url: "/video/..."} once Wav2Lip finishes.
    """
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "ready":
        video_id = os.path.basename(job["video_path"])
        return VideoJobStatus(ready=True, video_url=f"/video/{video_id}")
    if job["status"] == "error":
        return VideoJobStatus(ready=False, error=job.get("error"))
    return VideoJobStatus(ready=False)


@app.get("/video/{video_id}")
def stream_video(video_id: str, request: Request):
    """
    Serve the generated lip-sync video back to the LMS frontend.
    Supports HTTP Range requests so browsers can stream/seek the video.
    """
    video_path = f"/tmp/{video_id}"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found.")

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        try:
            range_val = range_header.strip().replace("bytes=", "")
            start_str, _, end_str = range_val.partition("-")
            start = int(start_str) if start_str else 0
            end   = int(end_str)   if end_str   else file_size - 1
        except ValueError:
            raise HTTPException(status_code=416, detail="Invalid Range header.")

        if start > end or end >= file_size:
            raise HTTPException(
                status_code=416,
                detail="Requested range not satisfiable.",
                headers={"Content-Range": f"bytes */{file_size}"},
            )

        chunk_size = end - start + 1

        def iterfile():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iterfile(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(chunk_size),
                "Accept-Ranges":  "bytes",
            },
        )

    # No Range header — serve the whole file
    def iterfile_full():
        with open(video_path, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                yield data

    return StreamingResponse(
        iterfile_full(),
        media_type="video/mp4",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges":  "bytes",
        },
    )


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    """
    Called by the LMS when the employee closes the Avatar widget or logs out.
    """
    delete_session(session_id)
    return {"message": "Session ended."}


class WelcomeRequest(BaseModel):
    session_id: str


@app.post("/session/welcome", response_model=ChatResponse)
def session_welcome(request: WelcomeRequest):
    """
    Returns the pre-rendered welcome video immediately — no TTS or Wav2Lip work
    at request time. The LMS profile is already loaded into the agent via
    /session/start so the conversation context is fully personalised from turn 1.
    """
    session = _require_session(request.session_id)
    logger.info(f"/session/welcome | session: {request.session_id}")
    return ChatResponse(
        session_id=request.session_id,
        reply=_WELCOME_GREETING,
        video_url="/assets/welcome.mp4",
    )


# ── Document ingestion ─────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    local_path: Optional[str] = "./hr_docs"
    azure_container: Optional[str] = None
    azure_connection_string: Optional[str] = None


@app.post("/admin/ingest")
def trigger_ingest(
    request: IngestRequest,
    authorization: str = Header(...),
):
    """
    Trigger RAG document ingestion from local folder and/or Azure Blob Storage.
    Protected by the same shared secret as /session/start.

    Body (all fields optional):
      local_path             — local folder to scan (default: ./hr_docs)
      azure_container        — Azure Blob Storage container name
      azure_connection_string — overrides AZURE_STORAGE_CONNECTION_STRING env var

    Example — ingest local only:
      POST /admin/ingest  {}

    Example — ingest Azure only:
      POST /admin/ingest  {"local_path": null, "azure_container": "hr-documents"}

    Example — ingest both:
      POST /admin/ingest  {"local_path": "./hr_docs", "azure_container": "hr-documents"}
    """
    if not authorization.startswith("Bearer ") or authorization[7:] != _SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing shared secret.")

    rag = RAGManager()
    try:
        total = rag.ingest_all(
            local_path=request.local_path,
            azure_container=request.azure_container,
            azure_connection_string=request.azure_connection_string,
        )
    except Exception as e:
        logger.error(f"/admin/ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"/admin/ingest complete | chunks ingested: {total}")
    return {"chunks_ingested": total, "status": "ok"}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
