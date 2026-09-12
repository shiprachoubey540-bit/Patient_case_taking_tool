"""
MediKiosk — Patient Kiosk backend.

This module does NOT re-implement intake, OCR, or speech-to-text. It's a
thin orchestration layer in front of three independent services
(converse-module on CONVERSE_URL, ocr-module on OCR_URL, voice-module on
VOICE_URL — default localhost:8000 / :8001 / :8003):

  - creates one "visit" per patient check-in, and ties together the
    Converse-module session_id and the OCR-module session_id (they're kept
    as separate concepts internally, since that's how the services already
    work — the kiosk just makes sure both get the same visit id)
  - proxies conversation turns, document uploads, and voice-clip
    transcription through to the right service
  - exposes one GET /visits/{id}/summary that pulls the intake + documents
    together into what the confirmation screen needs, so the frontend isn't
    juggling multiple origins and merge logic itself

Explicitly OUT of scope here:
  - the actual intake questions / extraction logic — those live in the two
    services this proxies to
  - auth / patient identity verification — patient_name is a free-text
    display label typed at the kiosk, not a verified identity. A real
    deployment would need ABHA/patient-ID lookup here; deliberately not
    faked for a demo. The doctor-facing lookup below (by `token`) has the
    SAME limitation one level up: token identifies a visit THIS kiosk
    created, not a verified patient — see store.py's docstring.
  - persistence for converse-module/ocr-module's OWN live session state —
    see db.py's module docstring for exactly what this module's own
    persistence layer does and doesn't cover. Short version: a kiosk-backend
    restart is now survivable; a converse-module/ocr-module restart
    mid-interview still isn't.
  - auth on the doctor-facing endpoints/page below — same "not faked for a
    demo" stance as patient identity. Anyone who can reach this service can
    reach /doctor.html and every /doctor/... endpoint. Fine for a kiosk on a
    trusted local network during a pilot; NOT fine to expose past that
    without adding real staff auth first.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import clients, db, store

# Must happen before clients.py's converse_url()/ocr_url() read os.environ —
# a missing load_dotenv() call has already bitten this project once (the
# Converse module silently never read its .env until this was added there).
load_dotenv()

# Mirrors converse-module/app/main.py's _TRIAGE_ORDER (that module owns the
# real definition — this is a display-ordering copy for the doctor queue,
# not clinical logic of its own). Keep in sync if that list ever changes;
# an unrecognized/null value sorts last rather than raising, since a visit
# with no triage_level yet (first turn, or a non-Ollama/Gemini backend
# quirk) should still show up in the queue, just at the bottom.
_TRIAGE_ORDER = {"emergency": 0, "urgent": 1, "soon": 2, "routine": 3}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    loaded = store.rehydrate_from_db()
    print(f"[kiosk] loaded {loaded} visit(s) from {db.db_path()}")
    yield


app = FastAPI(title="MediKiosk — Patient Kiosk", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_BACKENDS = {"auto", "gemini", "ollama", "mock"}


def ocr_backend_for(visit: store.Visit) -> str:
    """The OCR module has no vision-capable Ollama backend wired up (see its
    README — noted as a future option, not built). If the visit was started
    with "ollama", silently sending that to OCR would just 400. Rather than
    surface that as a confusing failure on the upload screen, fall back to
    "auto" for OCR calls specifically and say so via the /visits/{id} status
    the frontend already reads."""
    return "auto" if visit.backend == "ollama" else visit.backend


class StartVisitRequest(BaseModel):
    patient_name: Optional[str] = None
    ayush_mode: bool = False
    backend: str = "auto"
    language: str = "en"  # interview language, e.g. "en" | "hi" — validated by the Converse module itself


class MessageRequest(BaseModel):
    text: str


@app.get("/backends")
async def list_backends():
    """Merged availability from both downstream services, so the kiosk's
    check-in screen can grey out a backend option without a patient hitting
    a failure mid-conversation. If a service is unreachable entirely, its
    section says so rather than the whole endpoint failing — the check-in
    screen should still work with whatever IS up."""
    try:
        converse = await clients.converse_backends()
        converse_error = None
    except HTTPException as exc:
        converse, converse_error = None, exc.detail

    try:
        ocr = await clients.ocr_backends()
        ocr_error = None
    except HTTPException as exc:
        ocr, ocr_error = None, exc.detail

    try:
        voice = await clients.voice_backends()
        voice_error = None
    except HTTPException as exc:
        voice, voice_error = None, exc.detail

    return {
        "converse": converse or {"error": converse_error},
        "ocr": ocr or {"error": ocr_error},
        "voice": voice or {"error": voice_error},
    }


async def _build_summary(visit: store.Visit) -> dict:
    """Everything the confirmation screen (and now the doctor lookup view)
    needs in one call: the visit's own metadata, the structured intake
    history, and whatever documents were scanned. Intake is the core of
    this module and its failure is surfaced normally (a patient can't
    confirm a history that failed to load). The document scan is optional —
    a patient may not have uploaded anything, or the OCR service may be down
    while intake still works — so a connection failure there degrades to an
    empty list with a note instead of blocking the whole confirmation
    screen. Factored out of the /summary endpoint (2026-09-11) so
    _persist_snapshot below can build the exact same shape without
    duplicating this logic."""
    history = await clients.converse_get_history(visit.converse_session_id)

    try:
        documents = await clients.ocr_list_documents(visit.id)
        timeline = await clients.ocr_timeline(visit.id)
        documents_error = None
    except HTTPException as exc:
        documents, timeline, documents_error = [], [], exc.detail

    return {
        "visit": visit,
        "history": history,
        "documents": documents,
        "timeline": timeline,
        "documents_error": documents_error,
    }


async def _persist_snapshot(visit: store.Visit) -> dict:
    """Builds the current summary and writes it to db.py, so staff can see
    the latest known state of a visit even across a kiosk-backend restart —
    see db.py's module docstring for exactly what that does and doesn't
    cover. Called after every mutation (visit creation, each message, each
    document upload), not just at completion — an interview that's only
    half-done when the process dies is exactly the case this exists for.
    Also updates visit.complete/red_flag/triage_level in the in-memory
    store, since main.py is the only place that ever learns those values
    (store.py itself only tracks what it's told). Returns the summary it
    built (or None if persistence failed — see below), so callers that
    already need it (start_visit, post_message, upload_document) don't have
    to build it twice.

    BEST-EFFORT ON PURPOSE: every exception here is caught and logged, never
    raised. Without this, a real bug would follow: the primary action
    (post a message, upload a document) can succeed and then this
    AFTERWARD step — its own separate calls to converse-module/ocr-module —
    fails for an unrelated reason (a transient network blip, one of those
    services being momentarily slow), and the exception would propagate up
    through post_message/upload_document and turn an already-successful
    patient action into a 502 the patient sees. Persisting a snapshot is
    valuable but strictly secondary to the interview actually working;
    it must never be able to break the thing it's recording."""
    try:
        summary = await _build_summary(visit)
        history_resp = summary["history"]
        visit.complete = bool(history_resp.get("complete", False))
        visit.red_flag = bool(history_resp.get("red_flag", False))
        visit.triage_level = history_resp.get("triage_level")
        db.save_snapshot(
            visit_id=visit.id,
            token=visit.token,
            patient_name=visit.patient_name,
            ayush_mode=visit.ayush_mode,
            backend=visit.backend,
            converse_session_id=visit.converse_session_id,
            converse_backend=visit.converse_backend,
            language=visit.language,
            created_at=visit.created_at,
            updated_at=time.time(),
            complete=visit.complete,
            red_flag=visit.red_flag,
            triage_level=visit.triage_level,
            summary=jsonable_summary(summary),
        )
        return summary
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"[kiosk] failed to persist a snapshot for visit {visit.id} (non-fatal): {exc!r}")
        return None


def jsonable_summary(summary: dict) -> dict:
    """summary["visit"] is a store.Visit (a pydantic model) — everything
    else in the dict is already plain dict/list/str data straight from
    httpx/JSON. json.dumps() in db.py can't serialize a pydantic model
    directly, so this converts just that one field. Kept as a small
    top-level function rather than inlined so it's obvious this exists
    because of that one non-JSON-native field, not because the whole
    summary needs special handling."""
    return {**summary, "visit": summary["visit"].model_dump()}


@app.post("/visits")
async def start_visit(req: StartVisitRequest, background_tasks: BackgroundTasks):
    if req.backend not in VALID_BACKENDS:
        raise HTTPException(400, f"backend must be one of {sorted(VALID_BACKENDS)}")

    session = await clients.converse_start_session(req.ayush_mode, req.backend, req.language)
    visit = store.create_visit(
        patient_name=req.patient_name,
        ayush_mode=req.ayush_mode,
        backend=req.backend,
        converse_session_id=session["session_id"],
        converse_backend=session["backend"],
        language=req.language,
    )
    # Persistence is deliberately a background task, not awaited inline —
    # see _persist_snapshot's own docstring for why it's best-effort. It
    # used to be `await`ed here, which meant every patient action paid for
    # 3-4 extra network round-trips (a fresh history fetch, two OCR calls
    # that can't possibly find anything on a brand-new visit) before the
    # patient saw anything happen — that was the actual cause of "Begin
    # check-in takes a while" / "it all seems slow", not the LLM call.
    # Scheduling it here means the patient gets their response as soon as
    # the real work (starting the interview) is done; the snapshot catches
    # up moments later, invisible to them.
    background_tasks.add_task(_persist_snapshot, visit)
    return {
        "visit_id": visit.id,
        "token": visit.token,
        "question": session["question"],
        "backend": visit.backend,
        "converse_backend": visit.converse_backend,
        "ocr_backend_note": (
            f"document scan will use '{ocr_backend_for(visit)}' — Ollama has no vision backend wired up yet"
            if visit.backend == "ollama"
            else None
        ),
    }


def _get_visit(visit_id: str) -> store.Visit:
    try:
        return store.get_visit(visit_id)
    except KeyError:
        raise HTTPException(404, "visit not found")


@app.get("/visits/{visit_id}")
async def get_visit(visit_id: str):
    return _get_visit(visit_id)


@app.post("/visits/{visit_id}/message")
async def post_message(visit_id: str, req: MessageRequest, background_tasks: BackgroundTasks):
    visit = _get_visit(visit_id)
    result = await clients.converse_post_message(visit.converse_session_id, req.text)
    # See start_visit's comment: persistence runs after the response goes
    # back to the patient, not before — this is the call that used to add
    # 3 extra round-trips to every single chat turn.
    background_tasks.add_task(_persist_snapshot, visit)
    return result


@app.post("/visits/{visit_id}/message/stream")
async def post_message_stream(visit_id: str, req: MessageRequest):
    """SSE proxy for the "live tokens" feature — see
    clients.converse_post_message_stream and converse-module/app/main.py's
    post_message_stream for the actual protocol; this endpoint just wires a
    visit_id to the right converse-module session_id and relays bytes.

    Persistence runs via StreamingResponse's `background` hook instead of
    the injected BackgroundTasks dependency used elsewhere in this file —
    the effect is the same (best-effort, non-blocking, see
    _persist_snapshot's docstring), but a StreamingResponse's body isn't
    fully sent until the stream ends, and `background` is what guarantees
    this runs AFTER that, not the moment the response object is created.
    """
    visit = _get_visit(visit_id)
    return StreamingResponse(
        clients.converse_post_message_stream(visit.converse_session_id, req.text),
        media_type="text/event-stream",
        background=BackgroundTask(_persist_snapshot, visit),
    )


@app.get("/visits/{visit_id}/history")
async def get_history(visit_id: str):
    visit = _get_visit(visit_id)
    return await clients.converse_get_history(visit.converse_session_id)


@app.post("/visits/{visit_id}/transcribe")
async def transcribe_audio(
    visit_id: str,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
):
    """Speech-to-text only — this does NOT post the result as a message.
    The frontend fills the chat input box with the returned text so the
    patient (or a staff member) can review/correct it before it's sent,
    rather than auto-submitting whatever Whisper thought it heard straight
    into the clinical record. visit_id is only used to 404 consistently
    with the other endpoints — the voice module itself is stateless and
    doesn't know about visits."""
    _get_visit(visit_id)
    raw = await file.read()
    return await clients.voice_transcribe(
        audio_bytes=raw,
        filename=file.filename or "clip.webm",
        content_type=file.content_type or "audio/webm",
        language=language,
    )


@app.post("/visits/{visit_id}/documents")
async def upload_document(
    visit_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type_hint: Optional[str] = Form(default=None),
):
    visit = _get_visit(visit_id)
    raw = await file.read()
    result = await clients.ocr_upload(
        file_bytes=raw,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        document_type_hint=document_type_hint,
        session_id=visit.id,
        backend=ocr_backend_for(visit),
    )
    store.bump_document_count(visit_id)
    # Same as start_visit/post_message — don't make the patient wait on it.
    background_tasks.add_task(_persist_snapshot, visit)
    return result


@app.get("/visits/{visit_id}/documents")
async def list_documents(visit_id: str):
    visit = _get_visit(visit_id)
    return await clients.ocr_list_documents(visit.id)


@app.get("/visits/{visit_id}/timeline")
async def get_timeline(visit_id: str):
    visit = _get_visit(visit_id)
    return await clients.ocr_timeline(visit.id)


@app.get("/visits/{visit_id}/summary")
async def get_summary(visit_id: str):
    visit = _get_visit(visit_id)
    return await _build_summary(visit)


# ---- Doctor-facing view (2026-09-11) ---------------------------------------
#
# NO AUTH — see this module's docstring. Read-only: staff can look up and
# browse visits here, but there is deliberately no endpoint that lets this
# view edit clinical data; if a correction is needed it should happen through
# the same intake flow (or eventually a real EMR), not a side door here.

def _visit_queue_row(row: dict) -> dict:
    """Shapes one db.list_recent() row for the queue view — deliberately
    thinner than a full summary (no history/documents payload) so the queue
    endpoint stays cheap to poll even as more visits accumulate. The doctor
    page follows up with GET /doctor/visits/{token} for the one visit
    someone actually opens."""
    return {
        "id": row["id"],
        "token": row["token"],
        "patient_name": row["patient_name"],
        "chief_complaint": row["chief_complaint"],
        "ayush_mode": row["ayush_mode"],
        "backend": row["backend"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "complete": row["complete"],
        "red_flag": row["red_flag"],
        "triage_level": row["triage_level"],
    }


@app.get("/doctor/visits")
async def doctor_list_visits(limit: int = 100):
    """The queue view: most urgent first (red_flag, then triage_level, per
    _TRIAGE_ORDER above), most recently updated first within the same
    urgency — so a just-answered "routine" turn doesn't jump ahead of an
    "urgent" one from ten minutes ago just because it's fresher. Reads from
    db.py, not store.py's in-memory dict — that's deliberate: this should
    show every visit this kiosk has ever recorded, including ones from
    before the current process started, not just what's in memory right
    now."""
    rows = db.list_recent(limit=limit)
    rows.sort(
        key=lambda r: (
            0 if r["red_flag"] else 1,
            _TRIAGE_ORDER.get(r["triage_level"], 99),
            -r["updated_at"],
        )
    )
    return [_visit_queue_row(r) for r in rows]


@app.get("/doctor/visits/{token}")
async def doctor_get_visit(token: str):
    """Full detail for one visit, by its patient-facing token (see
    store.py's docstring — NOT a verified patient identity, just this
    kiosk's own lookup code). Prefers the LIVE summary (a fresh call to
    converse-module/ocr-module) when the visit is still in this process's
    memory, since that reflects anything that happened since the last
    snapshot; falls back to the last persisted snapshot in db.py otherwise
    — e.g. after a kiosk-backend restart, or once converse-module's own
    session has expired/gone. Either way the caller gets the best data
    available, and `live: false` in the response says which one it got so
    the UI can be honest about it rather than presenting a possibly-stale
    snapshot as current."""
    visit = store.get_visit_by_token(token)
    if visit is not None:
        try:
            summary = await _build_summary(visit)
            return {"live": True, **jsonable_summary(summary)}
        except HTTPException:
            pass  # converse-module unreachable/session gone — fall through to the snapshot below

    row = db.get_by_token(token)
    if row is None:
        raise HTTPException(404, "no visit found for that code")
    return {"live": False, **row["summary"]}


_templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

# The patient check-in flow is one page (templates/index.html) that {%
# include %}s five separate per-screen template files under
# templates/screens/ — reorganized 2026-09-11 for readability, purely
# organizational. The JS still does client-side show/hide between screens
# with no page reload, so this changes nothing about behavior: Jinja just
# assembles the same five sections into one page at request time instead of
# them being typed inline in one big HTML file. See templates/index.html's
# own comment for the same note. doctor.html lives alongside it in
# templates/ for consistency, even though (being one page already) it
# didn't need splitting.


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/doctor.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_doctor(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "doctor.html")
