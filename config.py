# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Voice sample
VOICE_SAMPLE = os.path.join(ASSETS_DIR, "hr_voice_sample.wav")
AVATAR_IMAGE = os.path.join(ASSETS_DIR, "hr_avatar.jpg")
AVATAR_SILENT_VIDEO = os.path.join(ASSETS_DIR, "hr_avatar_silent.mp4")

# LLM
OLLAMA_MODEL = "qwen3:4b"
EMBEDDING_MODEL = "nomic-embed-text"

# API endpoints (set via environment or default)
RECOMMENDATION_API_URL = os.getenv(
    "RECOMMENDATION_API_URL", "http://localhost:8001/recommend"
)
ASSESSMENT_API_URL = os.getenv("ASSESSMENT_API_URL", "http://localhost:8002/generate")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
