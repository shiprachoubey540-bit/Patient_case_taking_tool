"""
Durable visit storage (2026-09-11).

Added when persistence + a doctor-facing lookup view were requested for
real, on top of everything that already existed as in-memory-only. Scope,
stated plainly because it's easy to overclaim here:

WHAT THIS ACTUALLY PROTECTS: a kiosk-backend restart. Before this file
existed, store.py's docstring said outright that a restart loses every
visit — that's now only true for state this module doesn't know about.
Every meaningful mutation (visit created, a message answered, a document
uploaded) writes a full snapshot here, so a restarted kiosk-backend can
rehydrate its in-memory store.py._visits from this file and both (a) keep
routing an in-progress visit to the right converse/OCR session ids, and
(b) let staff look up ANY visit, complete or not, by its token.

WHAT THIS DOES NOT PROTECT: converse-module or ocr-module themselves
restarting. Those two services keep their own session state in their own
memory (converse-module/app/session_store.py, and ocr-module's equivalent)
— this file stores a snapshot of what they last reported, not a way to
resume talking to them. If either of those restarts mid-interview, the
patient's LIVE conversation breaks exactly as it did before this file
existed; the difference is that whatever was captured up to that point is
still visible to staff afterward instead of vanishing outright. Fixing that
properly means giving those two services their own persistence, which is a
larger, separate change this task didn't take on — flagging it as a real
gap, not pretending this file closes it.

WHY SQLITE, WHY NOT A REAL DATABASE: this is a single-kiosk demo/pilot
scale (one process, one machine, patients arrive one at a time) — sqlite3
is stdlib (no new dependency, nothing extra to install on legion), a single
file is trivial to inspect/back up/delete, and nothing here needs a
client-server database's concurrency guarantees. If this ever runs multiple
kiosks against shared state, that assumption stops holding and this should
become a real database — don't scale this file's pattern past that point
without revisiting it.

CONCURRENCY NOTE: a fresh connection is opened and closed per call rather
than one shared connection reused across requests — simpler to reason about
correctly than sharing a sqlite3.Connection across FastAPI's async/threaded
request handling, and fast enough at this scale that the extra connect()
overhead is irrelevant. WAL mode is enabled so reads (the doctor view)
don't block on a concurrent write (a patient's message being saved).
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

# Overridable so tests can point at a throwaway file instead of the real
# data/medikiosk.db — see this file's module docstring for why NOT ":memory:"
# (a fresh connection per call means ":memory:" would give every call its
# own empty database, not a shared one).
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "medikiosk.db"


def _db_path() -> Path:
    override = os.environ.get("MEDIKIOSK_DB_PATH")
    return Path(override) if override else _DEFAULT_DB_PATH


def db_path() -> Path:
    """Public accessor for callers outside this module (main.py logs it at
    startup) — the leading-underscore version above stays internal since
    everything else here can just call it directly without going through
    another layer."""
    return _db_path()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Call once at startup (main.py's lifespan). Safe to call repeatedly —
    CREATE TABLE IF NOT EXISTS, no destructive migration logic. If you add a
    column later, add it by hand here with an ALTER TABLE ... guarded the
    same way, rather than dropping/recreating — this file is meant to
    survive exactly the kind of restart a schema change happens across."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visits (
                id TEXT PRIMARY KEY,
                token TEXT UNIQUE NOT NULL,
                patient_name TEXT,
                ayush_mode INTEGER NOT NULL,
                backend TEXT NOT NULL,
                converse_session_id TEXT NOT NULL,
                converse_backend TEXT NOT NULL,
                language TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                complete INTEGER NOT NULL DEFAULT 0,
                red_flag INTEGER NOT NULL DEFAULT 0,
                triage_level TEXT,
                chief_complaint TEXT,
                summary_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_updated_at ON visits(updated_at)")
        # Denormalized from summary_json purely so the queue view (list_recent)
        # can show what a patient actually came in for without paying to
        # deserialize every row's full summary just to read one field.
        # Guarded try/except rather than a version check: sqlite's ALTER
        # TABLE has no "ADD COLUMN IF NOT EXISTS", and this needs to be a
        # no-op on every startup after the column already exists once.
        try:
            conn.execute("ALTER TABLE visits ADD COLUMN chief_complaint TEXT")
        except sqlite3.OperationalError:
            pass  # already exists — expected on every startup but the first


def save_snapshot(
    *,
    visit_id: str,
    token: str,
    patient_name: Optional[str],
    ayush_mode: bool,
    backend: str,
    converse_session_id: str,
    converse_backend: str,
    language: str,
    created_at: float,
    updated_at: float,
    complete: bool,
    red_flag: bool,
    triage_level: Optional[str],
    summary: dict,
) -> None:
    """Upsert — called after every meaningful mutation (visit start, each
    message, each document upload), not just at the end. That's deliberate:
    an interview that's only half-done when the process dies is exactly the
    case this exists for, so there's no "save at completion" shortcut here."""
    chief_complaint = (
        summary.get("history", {}).get("history", {}).get("chief_complaint")
        if isinstance(summary.get("history"), dict)
        else None
    )
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO visits (
                id, token, patient_name, ayush_mode, backend, converse_session_id,
                converse_backend, language, created_at, updated_at, complete,
                red_flag, triage_level, chief_complaint, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                patient_name=excluded.patient_name,
                updated_at=excluded.updated_at,
                complete=excluded.complete,
                red_flag=excluded.red_flag,
                triage_level=excluded.triage_level,
                chief_complaint=excluded.chief_complaint,
                summary_json=excluded.summary_json
            """,
            (
                visit_id, token, patient_name, int(ayush_mode), backend, converse_session_id,
                converse_backend, language, created_at, updated_at, int(complete),
                int(red_flag), triage_level, chief_complaint, json.dumps(summary),
            ),
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["ayush_mode"] = bool(d["ayush_mode"])
    d["complete"] = bool(d["complete"])
    d["red_flag"] = bool(d["red_flag"])
    d["summary"] = json.loads(d.pop("summary_json"))
    return d


def get_by_id(visit_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_by_token(token: str) -> Optional[dict]:
    """Case-insensitive on purpose — the token is meant to be read off a
    screen and typed in by a staff member under time pressure; rejecting a
    lowercase-typo of an otherwise-correct code would be a needless paper
    cut. See store.py for how the token itself is generated and why it
    avoids ambiguous characters for the same reason."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM visits WHERE token = ? COLLATE NOCASE", (token,)).fetchone()
    return _row_to_dict(row) if row else None


def list_recent(limit: int = 100) -> list[dict]:
    """For the doctor queue view and for rehydrating store.py's in-memory
    dict on startup. Returns metadata only (not the full summary_json blob —
    the queue view doesn't need it, and skipping it keeps this cheap even as
    the table grows); callers that need the full record should follow up
    with get_by_id/get_by_token for one visit at a time."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, token, patient_name, ayush_mode, backend, converse_session_id,
                   converse_backend, language, created_at, updated_at, complete,
                   red_flag, triage_level, chief_complaint
            FROM visits ORDER BY updated_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["ayush_mode"] = bool(d["ayush_mode"])
        d["complete"] = bool(d["complete"])
        d["red_flag"] = bool(d["red_flag"])
        out.append(d)
    return out
