# brain/session_context.py
# Thread-safe per-request storage for the active user's LMS profile.
# The web layer calls set_profile() before invoking the agent so that
# tools can read profile fields without asking the user for them.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_session_profile: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "session_profile", default=None
)


def set_profile(profile: Dict[str, Any]) -> None:
    """Called once per request before running the agent."""
    _session_profile.set(profile)


def get_profile() -> Optional[Dict[str, Any]]:
    """Read the profile that the LMS injected for this request."""
    return _session_profile.get()
