"""
In-memory session store — deliberately simple for a hackathon prototype.

Sessions vanish on server restart. That's fine for a demo. If this needs to
survive restarts (e.g. a multi-day pilot), swap this module for a real store
(Redis, Postgres) without touching main.py's calling code — that's the point
of keeping it behind create_session/get_session.
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from .schema import ClinicalHistory

MAX_TURNS = 14  # hard cap, independent of the model's own judgment


class Session:
    def __init__(self, ayush_mode: bool, backend: str, language: str = "en"):
        self.id = str(uuid.uuid4())
        self.ayush_mode = ayush_mode
        self.backend = backend  # "gemini" | "ollama" | "mock" — fixed for the life of the session
        self.language = language  # "en" | "hi" — fixed for the life of the session, see prompts.py
        self.transcript: List[dict] = []
        self.history = ClinicalHistory()
        self.complete = False
        self.red_flag = False
        self.red_flag_reason: Optional[str] = None
        self.triage_level: Optional[str] = None  # "routine" | "soon" | "urgent" | "emergency"
        self.created_at = time.time()


_SESSIONS: Dict[str, Session] = {}


def create_session(ayush_mode: bool, backend: str, language: str = "en") -> Session:
    session = Session(ayush_mode, backend, language)
    _SESSIONS[session.id] = session
    return session


def get_session(session_id: str) -> Session:
    if session_id not in _SESSIONS:
        raise KeyError(session_id)
    return _SESSIONS[session_id]
