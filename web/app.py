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

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from config import AVATAR_SILENT_VIDEO, ASSETS_DIR
from brain.session import create_session, delete_session, get_session, active_session_count
from transcriber.transcriber import Transcriber
from voice.voice import VoiceSynthesizer
from face.face import LipSyncGenerator
from logger import logger

# ── Shared heavy modules (loaded once at startup) ─────────────────────────────
transcriber = Transcriber()
voice = VoiceSynthesizer()
lipsync = LipSyncGenerator()

app = FastAPI(
    title="HR Avatar API",
    description="LMS-integrated conversational HR assistant",
    version="1.0.0",
)

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
    video_url: Optional[str] = None   # relative URL to stream the lip-sync video


# ── Helper ─────────────────────────────────────────────────────────────────────

def _require_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return session


def _run_avatar_pipeline(agent, text: str) -> tuple[str, str]:
    """
    Run agent → voice → lipsync.
    Returns (reply_text, video_path).
    """
    reply = agent.run(text)

    temp_voice = f"/tmp/voice_{uuid.uuid4().hex}.wav"
    voice.synthesize(reply, output_path=temp_voice)

    temp_video = f"/tmp/output_{uuid.uuid4().hex}.mp4"
    lipsync.generate(AVATAR_SILENT_VIDEO, temp_voice, temp_video)

    os.unlink(temp_voice)
    return reply, temp_video


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
    The LMS frontend sends the employee's typed or transcribed message.
    Returns the avatar's text reply and a URL to stream the lip-sync video.
    """
    session = _require_session(request.session_id)
    agent = session["agent"]

    logger.info(f"/chat | session: {request.session_id} | msg: {request.message[:80]}")

    reply, video_path = _run_avatar_pipeline(agent, request.message)

    # Expose the video via a one-time download endpoint
    video_id = os.path.basename(video_path)
    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        video_url=f"/video/{video_id}",
    )


@app.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(
    session_id: str,
    audio: UploadFile = File(..., description="WAV audio from the employee's microphone"),
):
    """
    Audio-based conversation turn.
    The LMS frontend uploads a WAV recording; the Avatar transcribes it,
    generates a reply, and returns the lip-sync video.
    """
    session = _require_session(session_id)
    agent = session["agent"]

    # Save uploaded audio temporarily
    temp_audio = f"/tmp/upload_{uuid.uuid4().hex}.wav"
    with open(temp_audio, "wb") as f:
        f.write(await audio.read())

    try:
        text = transcriber.transcribe(temp_audio)
        logger.info(f"/chat/audio | session: {session_id} | transcribed: {text[:80]}")
    finally:
        os.unlink(temp_audio)

    reply, video_path = _run_avatar_pipeline(agent, text)

    video_id = os.path.basename(video_path)
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        video_url=f"/video/{video_id}",
    )


@app.get("/video/{video_id}")
def stream_video(video_id: str):
    """
    Serve the generated lip-sync video back to the LMS frontend.
    Files are stored in /tmp and consumed once.
    """
    video_path = f"/tmp/{video_id}"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(video_path, media_type="video/mp4")


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    """
    Called by the LMS when the employee closes the Avatar widget or logs out.
    """
    delete_session(session_id)
    return {"message": "Session ended."}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
