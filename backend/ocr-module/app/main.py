"""
MediKiosk — Document OCR / Digitization module ("Scan" in the architecture
doc).

Scope of THIS module only:
  - takes a photo of a prior prescription/lab report/discharge summary
  - returns a structured, confidence-flagged extraction
  - keeps a simple chronological timeline across documents for one patient

Explicitly OUT of scope here (other modules/owners per the build plan):
  - the conversational history engine (see the separate converse-module)
  - FHIR/ABDM push — the backend/integration lead's module should consume
    GET /documents and GET /timeline as input to build a FHIR Bundle
  - drug-interaction checking — flagged as out of scope in the prompt too;
    that needs a real drug database, not something to fake for a demo
  - the real patient-facing kiosk UI — static/index.html here is a bare test
    harness (upload a file, see the extraction), not the deliverable
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import mock_ocr, store
from .schema import DOCUMENT_TYPES

load_dotenv()

app = FastAPI(title="MediKiosk — Document OCR / Digitization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_BACKENDS = {"auto", "gemini", "mock", "ollama"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12MB — generous for a phone photo, not for a batch dump


def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def ollama_available() -> bool:
    try:
        from . import ocr_ollama
        return ocr_ollama.is_available()
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    if requested == "auto":
        if gemini_available():
            return "gemini"
        elif ollama_available():
            return "ollama"
        else:
            return "mock"
    return requested


@app.get("/backends")
async def backends():
    return {
        "valid": sorted(VALID_BACKENDS - {"auto"}),
        "available": {
            "gemini": gemini_available(),
            "ollama": ollama_available(),
            "mock": True,
        },
    }


@app.post("/documents")
async def upload_document(
    backend: str = Form("auto"),
    document_type_hint: str | None = Form(None),
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    if backend not in VALID_BACKENDS:
        raise HTTPException(400, f"backend must be one of {sorted(VALID_BACKENDS)}")
    if document_type_hint is not None and document_type_hint not in DOCUMENT_TYPES:
        raise HTTPException(400, f"document_type_hint must be one of {DOCUMENT_TYPES}")

    resolved_backend = resolve_backend(backend)
    if resolved_backend == "gemini" and not gemini_available():
        raise HTTPException(400, "Gemini backend requested but GEMINI_API_KEY is not set (check your .env)")
    if resolved_backend == "ollama" and not ollama_available():
        raise HTTPException(400, "Ollama backend requested but Ollama is not reachable at localhost:11434")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"file too large ({len(raw)} bytes) — max {MAX_UPLOAD_BYTES} bytes")

    mime_type = file.content_type
    if not mime_type or not mime_type.startswith(("image/", "application/pdf")):
        raise HTTPException(400, "unsupported file type (must be image or pdf)")

    print(f"[{session_id}] document upload: {file.filename} ({len(raw)} bytes) -> {resolved_backend} engine")

    if resolved_backend == "mock":
        extract_fn = mock_ocr.extract
    elif resolved_backend == "ollama":
        from .ocr_ollama import extract as extract_fn
    else:
        from .ocr_llm import extract as extract_fn  # imported lazily so mock mode never needs google-genai configured

    try:
        extracted = extract_fn(raw, mime_type, document_type_hint)
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as a clean 502, not a crash
        raise HTTPException(502, f"extraction error: {exc}") from exc

    stored = store.save(extracted, resolved_backend, file.filename, session_id)
    return stored


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    try:
        return store.get(doc_id)
    except KeyError:
        raise HTTPException(404, "document not found")


@app.get("/documents")
def list_documents(session_id: str | None = None):
    return store.list_all(session_id)


@app.get("/timeline")
def get_timeline(session_id: str | None = None):
    entries = store.timeline(session_id)
    return [
        {
            "document_id": e["document"].id,
            "date_source": e["date_source"],
            "sort_date": e["sort_date"],
            "document_type": e["document"].extracted.document_type,
            "document_date": e["document"].extracted.document_date,
            "filename": e["document"].filename,
            "needs_review": e["document"].extracted.needs_review,
            "abnormal_investigations": [
                inv.model_dump() for inv in e["document"].extracted.investigations if inv.abnormal is True
            ],
        }
        for e in entries
    ]


_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
