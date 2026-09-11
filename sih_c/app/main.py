"""
Module C — Structured History Summary Generator (stub)

Calls Module B's /digitize endpoint for the uploaded document, combines
it with a placeholder conversation-history payload (standing in for
Module A until that teammate's real output format is ready), and
returns one merged case summary.

Run:
    uvicorn app.main:app --reload --port 8002
(Module B must already be running on port 8001)
"""
import json
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

from app.merge_logic import merge_case_summary
from app.html_view import render_case_summary_html

app = FastAPI(title="MediKiosk - Module C: Case Summary Generator (stub)")

MODULE_B_URL = "http://127.0.0.1:8001/digitize"

DEFAULT_CONVO = (
    '{"chief_complaint": "", "symptoms": [], "duration": "", '
    '"reported_medications": []}'
)


@app.get("/health")
def health():
    return {"status": "ok", "module": "C - Case Summary (stub)"}


async def _build_summary(file: UploadFile, conversation_history: str) -> dict:
    try:
        convo_dict = json.loads(conversation_history)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="conversation_history must be valid JSON")

    file_bytes = await file.read()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                MODULE_B_URL,
                files={"file": (file.filename, file_bytes, file.content_type)},
                timeout=60.0,
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=502,
                detail="Could not reach Module B at " + MODULE_B_URL +
                       " — make sure it's running on port 8001.",
            )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Module B error: {resp.text}")

    document_data = resp.json()
    return merge_case_summary(convo_dict, document_data)


@app.post("/case-summary")
async def build_case_summary(
    file: UploadFile = File(...),
    conversation_history: str = Form(
        default=DEFAULT_CONVO,
        description="JSON string — placeholder for Module A's output. "
                     "Paste real conversation data here to test merging.",
    ),
):
    """Machine-readable JSON — this is what Module D / a frontend would call."""
    return await _build_summary(file, conversation_history)


@app.post("/case-summary/view", response_class=HTMLResponse)
async def build_case_summary_view(
    file: UploadFile = File(...),
    conversation_history: str = Form(
        default=DEFAULT_CONVO,
        description="JSON string — placeholder for Module A's output.",
    ),
):
    """Human-readable HTML version of the same summary — for demos/judges."""
    summary = await _build_summary(file, conversation_history)
    return render_case_summary_html(summary)
