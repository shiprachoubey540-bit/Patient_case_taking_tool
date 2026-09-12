"""
MediKiosk — Conversational History Engine

Scope of THIS module only:
  - takes patient text turns (however they were produced — typed, or
    transcribed from speech by a separate ASR step) and runs the structured
    clinical interview
  - returns the next question plus the structured ClinicalHistory-so-far

Explicitly OUT of scope here (other modules/owners per the build plan):
  - speech-to-text / text-to-speech (voice lead wires that in front of
    POST /sessions/{id}/message — this module never touches audio)
  - document OCR
  - FHIR/ABDM push (backend/integration lead consumes GET .../history)
  - the real patient-facing kiosk UI (static/index.html here is a bare test
    harness, not the deliverable)
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file (GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_*) into
# the process environment. This MUST run before importing ollama_llm below —
# ollama_llm.py reads OLLAMA_URL/OLLAMA_MODEL/OLLAMA_NUM_CTX/OLLAMA_KEEP_ALIVE
# as module-level constants (`OLLAMA_MODEL = os.environ.get(...)`), evaluated
# once at import time. A load_dotenv() call placed AFTER that import (as this
# one used to be, further down this file) is too late: the import itself
# already read os.environ and froze those constants to their hardcoded
# defaults before .env ever got loaded. This was a real, previously-unnoticed
# bug — every OLLAMA_* override in .env (including OLLAMA_MODEL, meaning
# switching local models) was silently ignored, always falling back to
# gemma4:12b/8192/30m/localhost:11434, no matter what .env said. Caught
# 2026-09-11 when switching OLLAMA_MODEL to test a different local model had
# no effect even across full process restarts. GEMINI_MODEL doesn't have this
# problem because llm.py reads it inside a function at call time, not as a
# module-level constant — the safe pattern ollama_llm.py should have used too
# (not changed here, to keep this fix minimal; the import-order fix alone is
# enough, but worth knowing if this bites again for some OTHER module-level
# os.environ read added later).
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import mock_llm, ollama_llm, session_store, stream_extract
from .prompts import OPENING_QUESTIONS, RETRY_HINT, SUPPORTED_LANGUAGES
from .safety import keyword_red_flag_check
from .schema import ClinicalHistory, TriageLevel, TurnResult
from .turn_runner import FALLBACK_QUESTIONS, _is_empty_turn, run_turn_with_retry

_TRIAGE_ORDER: dict[TriageLevel, int] = {"routine": 0, "soon": 1, "urgent": 2, "emergency": 3}


def _apply_triage_level(session: session_store.Session, new_level) -> None:
    """Monotonic: a later turn can raise triage_level as more of the history
    comes out, but a momentary uncertain/lower-confidence turn can't silently
    walk it back down. Same latching philosophy as red_flag (session.red_flag
    is never unset once true) — the queue-facing signal should only ever get
    more cautious, not less, over the course of one interview."""
    if new_level is None:
        return
    if session.triage_level is None or _TRIAGE_ORDER[new_level] >= _TRIAGE_ORDER[session.triage_level]:
        session.triage_level = new_level

# (load_dotenv() moved to the top of this file's imports, 2026-09-11 — see
# the comment up there for why it had to move, not just why it exists.)

# Ollama cold-start warmup + keep-warm heartbeat (2026-09-07): see the
# "Cold-start warmup + heartbeat" note at the top of ollama_llm.py — a real
# patient on legion hit a 120s timeout on the very first message after a
# fresh start because nothing had loaded the model yet. lifespan() below
# fires a one-time warmup at server startup (in a background thread, so it
# never blocks uvicorn from accepting connections — a cold load can
# legitimately take a while and nobody should have to watch this process
# "hang" on boot for it), and _ollama_heartbeat_loop keeps re-warming it well
# inside OLLAMA_KEEP_ALIVE's window for as long as the server runs, so it
# never goes cold again during operating hours. Keep
# OLLAMA_HEARTBEAT_MINUTES comfortably below however OLLAMA_KEEP_ALIVE is
# set (default pairing: 20 min heartbeat / 30 min keep_alive) — a heartbeat
# slower than keep_alive just recreates the cold-start problem on a timer.
_HEARTBEAT_MINUTES = float(os.environ.get("OLLAMA_HEARTBEAT_MINUTES", "20"))
_heartbeat_stop = threading.Event()


def _ollama_heartbeat_loop() -> None:
    """Runs for the life of the process. Cheap when Ollama isn't in use at
    all (a Gemini/mock-only deployment pays only one is_available() check
    per interval) and re-warms it when it is, so a slow afternoon with big
    gaps between patients doesn't quietly let the model unload."""
    while not _heartbeat_stop.wait(_HEARTBEAT_MINUTES * 60):
        if ollama_llm.is_available():
            ollama_llm.warmup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ollama_llm.is_available():
        threading.Thread(target=ollama_llm.warmup, daemon=True).start()
    threading.Thread(target=_ollama_heartbeat_loop, daemon=True).start()
    yield
    _heartbeat_stop.set()


app = FastAPI(title="MediKiosk — Conversational History Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_BACKENDS = {"auto", "gemini", "ollama", "mock"}


def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def resolve_backend(requested: str) -> str:
    """"auto" prefers Ollama when it's reachable: local means no per-turn
    round trip to a third-party API for patient symptom text, and no
    dependency on internet connectivity in a clinic that may not have
    reliable access. Falls back to Gemini if Ollama isn't running, then mock
    if neither is available. Any explicit choice (gemini/ollama/mock) is
    returned as-is and validated by the caller — this ordering only applies
    to "auto"."""
    if requested == "auto":
        if ollama_llm.is_available():
            return "ollama"
        return "gemini" if gemini_available() else "mock"
    return requested


class StartRequest(BaseModel):
    ayush_mode: bool = False
    backend: str = "auto"  # "auto" | "gemini" | "ollama" | "mock"
    language: str = "en"  # see prompts.py's SUPPORTED_LANGUAGES for what's actually wired up


class MessageRequest(BaseModel):
    text: str


@app.get("/backends")
def list_backends():
    """Lets the UI grey out / warn about backends that aren't actually
    usable right now, instead of the user discovering that mid-interview."""
    return {
        "gemini": {
            "available": gemini_available(),
            "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        },
        "ollama": {
            "available": ollama_llm.is_available(),
            "model": ollama_llm.OLLAMA_MODEL,
            "url": ollama_llm.OLLAMA_URL,
        },
        "mock": {"available": True},
    }


@app.post("/sessions")
def start_session(req: StartRequest):
    if req.backend not in VALID_BACKENDS:
        raise HTTPException(400, f"backend must be one of {sorted(VALID_BACKENDS)}")
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")

    backend = resolve_backend(req.backend)

    if backend == "gemini" and not gemini_available():
        raise HTTPException(400, "Gemini backend requested but GEMINI_API_KEY is not set (check your .env)")
    if backend == "ollama" and not ollama_llm.is_available():
        raise HTTPException(
            400,
            f"Ollama backend requested but couldn't reach it at {ollama_llm.OLLAMA_URL} — "
            f"run `ollama serve` and `ollama pull {ollama_llm.OLLAMA_MODEL}` first",
        )

    session = session_store.create_session(req.ayush_mode, backend, req.language)
    opening_question = OPENING_QUESTIONS[req.language]
    session.transcript.append({"role": "assistant", "text": opening_question})
    return {
        "session_id": session.id,
        "language": session.language,
        "question": opening_question,
        "backend": backend,
    }


def _red_flag_response(session: session_store.Session, matched: str) -> dict:
    """Shared by both the plain and streaming message endpoints — the
    keyword safety net's response shape never depended on which endpoint
    caught it."""
    session.red_flag = True
    session.red_flag_reason = f"keyword match: '{matched}'"
    session.complete = True
    _apply_triage_level(session, "emergency")
    return {
        "acknowledgement": None,
        "question": None,
        "complete": True,
        "red_flag": True,
        "red_flag_reason": session.red_flag_reason,
        "triage_level": session.triage_level,
        "history_so_far": session.history.model_dump(),
        "backend": session.backend,
        "language": session.language,
    }


def _finalize_turn(session: session_store.Session, result: TurnResult, patient_turns: int) -> dict:
    """Applies a backend's TurnResult to the session and builds the response
    body — pulled out of post_message (2026-09-12) so the new streaming
    endpoint below can reach the exact same end state (and return the exact
    same final shape) without duplicating this logic. Nothing about the
    session-mutation semantics changed in this refactor."""
    # Local models (especially on the final 'complete=true' turn) often get lazy 
    # and omit the entire history block because they think they are done.
    # Instead of blindly overwriting session.history and wiping it clean, 
    # we do a deep merge: keep the old value if the new value is empty/None.
    old_h = session.history.model_dump()
    new_h = result.history.model_dump()
    
    # Merge top-level history fields
    for k, v in new_h.items():
        if k == "hpi" and isinstance(v, dict):
            old_hpi = old_h.get("hpi", {})
            for hk, hv in v.items():
                if hv:  # if new HPI field has a truthy value (not None/empty string/empty list)
                    old_hpi[hk] = hv
            old_h["hpi"] = old_hpi
        elif k == "ayush" and isinstance(v, dict):
            old_ayush = old_h.get("ayush", {}) or {}
            for ak, av in v.items():
                if av:
                    old_ayush[ak] = av
            old_h["ayush"] = old_ayush
        elif v:  # if new field has a truthy value
            old_h[k] = v
            
    session.history = ClinicalHistory.model_validate(old_h)

    if result.red_flag:
        session.red_flag = True
        session.red_flag_reason = result.red_flag_reason
    _apply_triage_level(session, result.triage_level)

    # MAX_TURNS counts patient answers, not raw transcript entries (which
    # double-count each Q+A exchange) — this was previously wrong and cut
    # the interview short after MAX_TURNS/2 answers.
    hard_cap_hit = patient_turns >= session_store.MAX_TURNS
    session.complete = result.complete or session.red_flag or hard_cap_hit

    if not session.complete and result.next_question:
        session.transcript.append({"role": "assistant", "text": result.next_question})

    return {
        "acknowledgement": result.acknowledgement,
        "question": None if session.complete else result.next_question,
        "complete": session.complete,
        "red_flag": session.red_flag,
        "red_flag_reason": session.red_flag_reason,
        "triage_level": session.triage_level,
        "history_so_far": session.history.model_dump(),
        "backend": session.backend,
        "language": session.language,
    }


def _turn_fn_for(backend: str):
    if backend == "mock":
        return mock_llm.run_turn
    if backend == "gemini":
        from .llm import run_turn as turn_fn  # imported lazily so other backends never need google-genai configured

        return turn_fn
    if backend == "ollama":
        return ollama_llm.run_turn
    raise HTTPException(500, f"session has unknown backend {backend!r}")


@app.post("/sessions/{session_id}/message")
def post_message(session_id: str, req: MessageRequest):
    try:
        session = session_store.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")

    if session.complete:
        # User clicked "Edit" after finishing, so re-open the session
        session.complete = False
        session.red_flag = False

    session.transcript.append({"role": "patient", "text": req.text})

    # Fast, rule-based safety net — runs before the LLM, so an emergency is
    # caught even if the model call fails or misjudges the turn.
    matched = keyword_red_flag_check(req.text)
    if matched:
        return _red_flag_response(session, matched)

    turn_fn = _turn_fn_for(session.backend)
    patient_turns = sum(1 for t in session.transcript if t["role"] == "patient")

    try:
        current_history_json = session.history.model_dump_json()
        result = run_turn_with_retry(turn_fn, session.transcript, session.ayush_mode, session.language, patient_turns, current_history_json)
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as a clean 502, not a crash
        raise HTTPException(502, f"history engine error: {exc}") from exc

    return _finalize_turn(session, result, patient_turns)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# The two TurnResult string fields worth showing the patient as they're
# generated, in the exact order they appear in the schema (see
# stream_extract.py's docstring for why order matters here). "history" and
# everything after "next_question" never gets streamed to the UI — the
# patient only ever needs to see what the assistant is "saying."
_STREAM_FIELDS = ["acknowledgement", "next_question"]


@app.post("/sessions/{session_id}/message/stream")
async def post_message_stream(session_id: str, req: MessageRequest):
    """SSE variant of post_message — the "live tokens" feature (2026-09-12):
    the patient sees the acknowledgement/next_question text appear
    progressively as the model generates it, instead of everything showing
    up at once after the full (often 10s-60s+ for a local model) turn
    completes.

    Event protocol (frontend contract):
      - `delta`   {"field": "acknowledgement"|"next_question", "text": str}
                  — append this text to that field's bubble.
      - `replace` {"acknowledgement": str, "next_question": str} — rare
                  (ollama only): the streamed first attempt was empty/invalid
                  (see schema.py's min_length=1 — should now be uncommon) and
                  got retried non-streamed; overwrite whatever partial text
                  was already shown with these final values.
      - `final`   the exact same JSON shape post_message returns — use it
                  for session state (complete/red_flag/triage/history), NOT
                  for text — that already arrived via delta/replace.
      - `error`   {"detail": str} — something failed; show it like any other
                  request error and let the patient retry.

    Only the ollama backend gets genuine incremental streaming (Ollama's
    stream=True /api/chat, piped through stream_extract's field extractor).
    gemini/mock run exactly as before and then emit their whole
    acknowledgement/next_question as single `delta` events — no perceived
    latency win for those, but it keeps the frontend's protocol uniform
    regardless of backend, which matters more than optimizing backends that
    weren't the ones patients were staring at a frozen screen for.
    """
    try:
        session = session_store.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")

    if session.complete:
        # User clicked "Edit" after finishing, so re-open the session
        session.complete = False
        session.red_flag = False

    session.transcript.append({"role": "patient", "text": req.text})

    matched = keyword_red_flag_check(req.text)
    if matched:
        final = _red_flag_response(session, matched)

        async def redflag_stream():
            yield _sse("final", final)

        return StreamingResponse(redflag_stream(), media_type="text/event-stream")

    patient_turns = sum(1 for t in session.transcript if t["role"] == "patient")

    async def event_stream():
        try:
            if session.backend == "ollama":
                extractor = stream_extract.IncrementalFieldExtractor(_STREAM_FIELDS)
                full_text_parts: list[str] = []
                current_history_json = session.history.model_dump_json()
                try:
                    async for chunk in ollama_llm.astream_turn(
                        session.transcript, session.ayush_mode, session.language, patient_turns, current_history=current_history_json
                    ):
                        full_text_parts.append(chunk)
                        for field, delta in extractor.feed(chunk):
                            yield _sse("delta", {"field": field, "text": delta})
                except RuntimeError as exc:
                    yield _sse("error", {"detail": str(exc)})
                    return

                try:
                    result = TurnResult.from_raw_lenient("".join(full_text_parts))
                except Exception:  # noqa: BLE001 - schema-invalid streamed output is handled like an empty turn below
                    result = None

                if result is None or _is_empty_turn(result):
                    # Rare now (schema.py's min_length=1 should mostly
                    # prevent this) — retry once, non-streamed, with the
                    # same correction the plain endpoint's retry uses, then
                    # tell the frontend to overwrite whatever partial text
                    # it already rendered from the failed first attempt.
                    try:
                        result = ollama_llm.run_turn(
                            session.transcript, session.ayush_mode, session.language, patient_turns, RETRY_HINT, current_history_json
                        )
                    except Exception as exc:  # noqa: BLE001 - same handling as post_message's 502 case
                        yield _sse("error", {"detail": f"history engine error: {exc}"})
                        return
                    if _is_empty_turn(result):
                        fallback = FALLBACK_QUESTIONS.get(session.language, FALLBACK_QUESTIONS["en"])
                        result = result.model_copy(update={"next_question": fallback})
                    yield _sse(
                        "replace",
                        {"acknowledgement": result.acknowledgement or "", "next_question": result.next_question or ""},
                    )
            else:
                try:
                    turn_fn = _turn_fn_for(session.backend)
                except HTTPException as exc:
                    yield _sse("error", {"detail": exc.detail})
                    return
                try:
                    current_history_json = session.history.model_dump_json()
                    result = run_turn_with_retry(
                        turn_fn, session.transcript, session.ayush_mode, session.language, patient_turns, current_history_json
                    )
                except Exception as exc:  # noqa: BLE001 - same handling as post_message's 502 case
                    yield _sse("error", {"detail": f"history engine error: {exc}"})
                    return
                if result.acknowledgement:
                    yield _sse("delta", {"field": "acknowledgement", "text": result.acknowledgement})
                if result.next_question:
                    yield _sse("delta", {"field": "next_question", "text": result.next_question})

            final = _finalize_turn(session, result, patient_turns)
            yield _sse("final", final)
        except Exception as exc:  # noqa: BLE001 - never let the SSE stream just die with no explanation
            yield _sse("error", {"detail": f"history engine error: {exc}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str):
    try:
        session = session_store.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    return {
        "session_id": session.id,
        "ayush_mode": session.ayush_mode,
        "backend": session.backend,
        "language": session.language,
        "complete": session.complete,
        "red_flag": session.red_flag,
        "red_flag_reason": session.red_flag_reason,
        "triage_level": session.triage_level,
        "history": session.history.model_dump(),
    }


_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
