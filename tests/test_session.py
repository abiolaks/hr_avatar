# tests/test_session.py
# Tests for the session store and /session/start + /chat endpoints.
# Uses mocks — no Ollama, ChromaDB, or XTTS needed.

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


_PROFILE = {
    "user_id": "emp_001",
    "name": "Abiola K.",
    "job_role": "Data Analyst",
    "department": "Engineering",
    "skill_level": "Intermediate",
    "known_skills": ["SQL", "Python"],
    "enrolled_courses": ["data-101"],
    "context": "avatar_chat",
}

_AUTH = "Bearer dev-secret"


# ── Session store unit tests ──────────────────────────────────────────────────

class TestSessionStore:

    def test_create_and_get_session(self):
        with patch("brain.session.HRAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            from brain.session import create_session, get_session
            sid = create_session(_PROFILE.copy())
            assert sid.startswith("sess_")
            session = get_session(sid)
            assert session is not None
            assert session["profile"]["user_id"] == "emp_001"

    def test_get_missing_session_returns_none(self):
        from brain.session import get_session
        assert get_session("sess_doesnotexist") is None

    def test_delete_session(self):
        with patch("brain.session.HRAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            from brain.session import create_session, get_session, delete_session
            sid = create_session(_PROFILE.copy())
            delete_session(sid)
            assert get_session(sid) is None


# ── API endpoint tests ────────────────────────────────────────────────────────

def _make_client():
    """Build a TestClient with all heavy modules mocked out."""
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "I want to learn machine learning"

    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = None

    mock_lipsync = MagicMock()
    mock_lipsync.generate.return_value = "/tmp/fake_video.mp4"

    mock_agent = MagicMock()
    mock_agent.run.return_value = "Here are some ML courses for you."

    patches = [
        patch("web.app.transcriber", mock_transcriber),
        patch("web.app.voice", mock_voice),
        patch("web.app.lipsync", mock_lipsync),
        patch("web.app.AVATAR_SILENT_VIDEO", "/tmp/fake_silent.mp4"),
    ]
    return patches, mock_agent


class TestSessionStartEndpoint:

    def test_valid_request_returns_session_id(self):
        with patch("brain.session.HRAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            from web.app import app
            client = TestClient(app)
            response = client.post(
                "/session/start",
                json=_PROFILE,
                headers={"Authorization": _AUTH},
            )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"].startswith("sess_")

    def test_missing_auth_returns_401(self):
        from web.app import app
        client = TestClient(app)
        response = client.post("/session/start", json=_PROFILE)
        assert response.status_code == 422  # missing required header

    def test_wrong_secret_returns_401(self):
        with patch("brain.session.HRAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            from web.app import app
            client = TestClient(app)
            response = client.post(
                "/session/start",
                json=_PROFILE,
                headers={"Authorization": "Bearer wrong-secret"},
            )
        assert response.status_code == 401


class TestChatEndpoint:

    def test_chat_returns_reply(self):
        with patch("brain.session.HRAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run.return_value = "Here are your recommendations."
            MockAgent.return_value = mock_agent_instance

            import os, unittest.mock as um
            with um.patch("os.path.exists", return_value=True), \
                 um.patch("os.unlink"), \
                 um.patch("web.app.voice") as mv, \
                 um.patch("web.app.lipsync") as ml:
                mv.synthesize.return_value = None
                ml.generate.return_value = "/tmp/fake.mp4"

                from web.app import app
                client = TestClient(app)

                # Start a session first
                start = client.post(
                    "/session/start",
                    json=_PROFILE,
                    headers={"Authorization": _AUTH},
                )
                session_id = start.json()["session_id"]

                response = client.post(
                    "/chat",
                    json={"session_id": session_id, "message": "I want to learn ML"},
                )

            assert response.status_code == 200
            data = response.json()
            assert "reply" in data
            assert data["reply"] == "Here are your recommendations."

    def test_chat_invalid_session_returns_404(self):
        from web.app import app
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"session_id": "sess_invalid", "message": "hello"},
        )
        assert response.status_code == 404
