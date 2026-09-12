"""
In-memory visit store — now a fast-path cache in front of db.py, not the
only copy (2026-09-11).

A "visit" is the kiosk's own concept — it's what ties one patient's
conversational intake (a Converse-module session_id) and their uploaded
documents (grouped under the OCR module's session_id) together as a single
thing the frontend can drive through one screen flow. This dict is still
what every request path reads/writes during normal operation (it's just a
dict lookup, no disk I/O per request) — db.py is what makes that state
survive a restart. See db.py's module docstring for exactly what is and
isn't protected by that; don't assume "there's a database now" means every
restart scenario is covered, because it doesn't.

`token` (added 2026-09-11): a short, human-typeable code shown to the
patient at check-in and used by the doctor-facing lookup view — NOT a
verified patient identity. main.py's own docstring already says patient_name
is a free-text display label with no ABHA/identity check behind it; token is
the same kind of honest placeholder, one level up (a lookup key for THIS
kiosk's own record of a visit, not a real patient ID). Keep that distinction
in any UI copy that references it.
"""
from __future__ import annotations

import secrets
import time
import uuid
from typing import Dict, Optional

from pydantic import BaseModel, Field

from . import db

# Deliberately excludes visually-ambiguous characters (0/O, 1/I/L) since
# this is read off a small screen and typed in by a staff member, possibly
# in a hurry — a code that's technically unique but hard to transcribe
# correctly is a paper cut waiting to happen. 6 chars from this 32-symbol
# alphabet is ~30 bits — collision odds are irrelevant at kiosk-demo scale,
# and generate_token() re-rolls on the (practically never hit) collision
# case anyway rather than assuming it away.
_TOKEN_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def _generate_token() -> str:
    return "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(6))


class Visit(BaseModel):
    id: str
    token: str
    patient_name: Optional[str] = None
    ayush_mode: bool = False
    # The backend the patient/staff picked at check-in, e.g. "auto" | "gemini" | "ollama" | "mock".
    # This is passed to the Converse module as-is. The OCR module doesn't support
    # "ollama" (no vision model wired up) — see main.py's ocr_backend_for() for
    # how that gap is handled rather than silently ignored.
    backend: str = "auto"
    language: str = "en"  # the interview language picked at check-in, e.g. "en" | "hi" — passed to Converse as-is
    converse_session_id: str
    converse_backend: str  # what the Converse module actually resolved "auto" to
    created_at: float = Field(default_factory=time.time)
    documents_uploaded: int = 0
    # Mirrors the LATEST values main.py has seen from Converse for this
    # visit (complete/red_flag/triage_level) — kept here too (duplicated
    # from the Converse-module response) purely so save_snapshot() below has
    # something to persist without re-fetching history on every call that
    # doesn't otherwise need it. main.py is what actually updates these.
    complete: bool = False
    red_flag: bool = False
    triage_level: Optional[str] = None


_visits: Dict[str, Visit] = {}
_visits_by_token: Dict[str, str] = {}  # token (uppercased) -> visit id


def create_visit(
    patient_name: Optional[str],
    ayush_mode: bool,
    backend: str,
    converse_session_id: str,
    converse_backend: str,
    language: str = "en",
) -> Visit:
    token = _generate_token()
    while token in _visits_by_token:  # practically unreachable at this scale; correctness over optimism
        token = _generate_token()
    visit = Visit(
        id=str(uuid.uuid4()),
        token=token,
        patient_name=patient_name,
        ayush_mode=ayush_mode,
        backend=backend,
        language=language,
        converse_session_id=converse_session_id,
        converse_backend=converse_backend,
    )
    _visits[visit.id] = visit
    _visits_by_token[token.upper()] = visit.id
    return visit


def get_visit(visit_id: str) -> Visit:
    return _visits[visit_id]  # raises KeyError — caller maps to 404


def get_visit_by_token(token: str) -> Optional[Visit]:
    """In-memory fast path only — a token from BEFORE the current process
    started won't be here even though it's still in db.py. main.py's doctor
    lookup endpoint falls back to db.py when this misses, so the caller
    doesn't need to know which path answered."""
    visit_id = _visits_by_token.get(token.strip().upper())
    return _visits.get(visit_id) if visit_id else None


def bump_document_count(visit_id: str) -> None:
    _visits[visit_id].documents_uploaded += 1


def rehydrate_from_db() -> int:
    """Called once at startup (main.py's lifespan), before this process has
    served any request. Reloads every visit db.py knows about into this
    in-memory cache so an in-progress visit keeps routing correctly across a
    kiosk-backend-only restart (see db.py's module docstring for the limits
    of that — this does NOT help if converse-module/ocr-module also
    restarted). Returns the count loaded, purely so main.py can log
    something meaningful at startup instead of silence."""
    rows = db.list_recent(limit=10_000)
    for row in rows:
        visit = Visit(
            id=row["id"],
            token=row["token"],
            patient_name=row["patient_name"],
            ayush_mode=row["ayush_mode"],
            backend=row["backend"],
            language=row["language"],
            converse_session_id=row["converse_session_id"],
            converse_backend=row["converse_backend"],
            created_at=row["created_at"],
            complete=row["complete"],
            red_flag=row["red_flag"],
            triage_level=row["triage_level"],
        )
        _visits[visit.id] = visit
        _visits_by_token[visit.token.upper()] = visit.id
    return len(rows)
