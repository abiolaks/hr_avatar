# brain/session.py
# In-memory session store.
# The LMS calls POST /session/start to create a session and gets back a
# session_id.  Every subsequent /chat request carries that session_id so
# the avatar can load the right user profile and conversation history.

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from brain.agent import HRAgent
from logger import logger

# Sessions expire after 1 hour of inactivity
SESSION_TTL_MINUTES = 60

# session_id → { profile, agent, created_at, last_active }
_store: Dict[str, Dict[str, Any]] = {}


def create_session(profile: Dict[str, Any]) -> str:
    """
    Store the LMS user profile and spin up a dedicated HRAgent for this
    session.  Returns the session_id the LMS frontend must forward on
    every /chat request.
    """
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    _store[session_id] = {
        "profile": profile,
        "agent": HRAgent(),
        "created_at": datetime.now(timezone.utc),
        "last_active": datetime.now(timezone.utc),
    }
    logger.info(f"Session created: {session_id} | user: {profile.get('user_id')}")
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Return session dict or None if missing / expired."""
    session = _store.get(session_id)
    if session is None:
        return None

    # Expire stale sessions
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
    if session["last_active"] < cutoff:
        delete_session(session_id)
        logger.info(f"Session expired: {session_id}")
        return None

    session["last_active"] = datetime.now(timezone.utc)
    return session


def delete_session(session_id: str) -> None:
    """Explicitly remove a session (e.g. employee logs out of LMS)."""
    _store.pop(session_id, None)
    logger.info(f"Session deleted: {session_id}")


def active_session_count() -> int:
    return len(_store)


def prune_expired_sessions() -> int:
    """
    Remove all sessions that have been inactive longer than SESSION_TTL_MINUTES.
    Called periodically by the web layer so abandoned sessions don't accumulate
    in memory indefinitely (lazy expiry alone only cleans up on access).
    Returns the number of sessions pruned.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
    # Compare aware datetimes — all timestamps now use timezone.utc
    expired = [sid for sid, s in _store.items() if s["last_active"] < cutoff]
    for sid in expired:
        _store.pop(sid, None)
    if expired:
        logger.info(f"Background prune: removed {len(expired)} expired session(s)")
    return len(expired)
