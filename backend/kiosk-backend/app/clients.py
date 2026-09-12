"""
Thin async HTTP clients for the independent services this module
orchestrates: the Converse (conversational intake) module, the OCR
(document scan) module, and the Voice (speech-to-text) module. Each is a
separate FastAPI process with its own port — this module does NOT import
their code, it talks to them over HTTP, same as a real deployment would
(they could be on different machines).

Every call goes through _request() so failure modes are consistent and
honest: if a downstream service is down, the kiosk says so clearly (502,
naming which service and its URL) instead of the frontend seeing a generic
network error or a silent hang.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import HTTPException

# 220s read timeout. This has to be longer than the SLOWEST thing any
# downstream service might legitimately still be doing, not just "long
# enough for the common case" — otherwise this proxy kills a request that
# was actually still working and reports it to the patient as a failure,
# when the truth is just "still thinking."
#
# Updated 2026-09-07: the binding constraint is now converse-module's own
# Ollama path timeout, raised 120s -> 200s
# (converse-module/app/ollama_llm.py's _RUN_TURN_TIMEOUT_S) after a real
# cold-start on legion blew through the old 120s before Ollama even
# responded — see that file's "Cold-start warmup + heartbeat" note for the
# full story and why a startup warmup + periodic heartbeat, not just a
# bigger number, is the actual fix. 220s here still needs to stay above
# converse-module's 200s with some margin (turn_runner.py's empty-turn retry
# means a slow-but-successful call there could in principle repeat once, so
# don't assume 200s is the hard ceiling on converse-module's side either).
# If Ollama is routinely taking anywhere near this long AFTER warmup has had
# a chance to run, that's worth knowing about on its own terms (check
# `ollama ps` — is it actually running on GPU? already confirmed yes on
# legion — or look at converse-module's `[llm/ollama] ... load_duration_s=`
# log line to see whether it's a reload or genuinely slow inference) rather
# than just raising this number further again.
#
# 10s connect timeout: if the service isn't even listening, fail fast
# rather than hang the kiosk UI.
_TIMEOUT = httpx.Timeout(220.0, connect=10.0)


def converse_url() -> str:
    return os.environ.get("CONVERSE_URL", "http://localhost:8000")


def ocr_url() -> str:
    return os.environ.get("OCR_URL", "http://localhost:8001")


def voice_url() -> str:
    return os.environ.get("VOICE_URL", "http://localhost:8003")


async def _request(method: str, base_url: str, path: str, service_name: str, **kwargs: Any):
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
            resp = await client.request(method, path, **kwargs)
    except httpx.ConnectError as exc:
        raise HTTPException(
            502,
            f"Could not reach the {service_name} at {base_url} — is it running? "
            f"(the kiosk backend only orchestrates; it doesn't start the other services for you)",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(504, f"{service_name} timed out") from exc

    if resp.status_code >= 400:
        detail = resp.text
        try:
            parsed = resp.json()
            detail = parsed.get("detail", detail)
        except ValueError:
            pass
        raise HTTPException(resp.status_code, f"{service_name}: {detail}")

    return resp.json()


# ---- Converse module -------------------------------------------------------

async def converse_backends() -> dict:
    return await _request("GET", converse_url(), "/backends", "Converse (intake) module")


async def converse_start_session(ayush_mode: bool, backend: str, language: str = "en") -> dict:
    return await _request(
        "POST", converse_url(), "/sessions", "Converse (intake) module",
        json={"ayush_mode": ayush_mode, "backend": backend, "language": language},
    )


async def converse_post_message(session_id: str, text: str) -> dict:
    return await _request(
        "POST", converse_url(), f"/sessions/{session_id}/message", "Converse (intake) module",
        json={"text": text},
    )


async def converse_post_message_stream(session_id: str, text: str) -> AsyncIterator[bytes]:
    """SSE proxy for the "live tokens" feature (2026-09-12) — relays
    converse-module's event stream through byte-for-byte rather than
    understanding it, so kiosk-backend doesn't need to know or care about
    the event protocol converse-module and the frontend agree on (see
    converse-module/app/main.py's post_message_stream docstring for that
    protocol).

    Deliberately does NOT raise HTTPException the way _request() does for
    the non-streaming calls above: by the time an error surfaces here, the
    HTTP response to the frontend has likely already started (status 200,
    headers sent) as part of turning this generator into a StreamingResponse
    — changing the status code at that point isn't possible. So connection
    problems are reported IN-BAND, as an SSE `error` event the frontend
    already knows how to handle (same event name and shape converse-module
    itself uses for its own errors), rather than as an exception the caller
    can't do anything useful with once streaming has begun.
    """
    url = f"{converse_url()}/sessions/{session_id}/message/stream"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream("POST", url, json={"text": text}) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    detail = body.decode(errors="replace")
                    try:
                        detail = json.loads(detail).get("detail", detail)
                    except ValueError:
                        pass
                    yield _sse_error(f"Converse (intake) module: {detail}")
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except httpx.ConnectError:
        yield _sse_error(
            f"Could not reach the Converse (intake) module at {converse_url()} — is it running? "
            f"(the kiosk backend only orchestrates; it doesn't start the other services for you)"
        )
    except httpx.TimeoutException:
        yield _sse_error("Converse (intake) module timed out")


def _sse_error(detail: str) -> bytes:
    return f"event: error\ndata: {json.dumps({'detail': detail})}\n\n".encode()


async def converse_get_history(session_id: str) -> dict:
    return await _request("GET", converse_url(), f"/sessions/{session_id}/history", "Converse (intake) module")


# ---- OCR module -------------------------------------------------------------

async def ocr_backends() -> dict:
    return await _request("GET", ocr_url(), "/backends", "OCR (document scan) module")


async def ocr_upload(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    document_type_hint: Optional[str],
    session_id: str,
    backend: str,
) -> dict:
    data: dict[str, str] = {"backend": backend, "session_id": session_id}
    if document_type_hint:
        data["document_type_hint"] = document_type_hint
    files = {"file": (filename, file_bytes, content_type or "application/octet-stream")}
    return await _request("POST", ocr_url(), "/documents", "OCR (document scan) module", data=data, files=files)


async def ocr_list_documents(session_id: str) -> list:
    return await _request("GET", ocr_url(), "/documents", "OCR (document scan) module", params={"session_id": session_id})


async def ocr_timeline(session_id: str) -> list:
    return await _request("GET", ocr_url(), "/timeline", "OCR (document scan) module", params={"session_id": session_id})


# ---- Voice module -----------------------------------------------------------

async def voice_backends() -> dict:
    return await _request("GET", voice_url(), "/backends", "Voice (speech-to-text) module")


async def voice_transcribe(audio_bytes: bytes, filename: str, content_type: str, language: Optional[str]) -> dict:
    data: dict[str, str] = {"backend": "auto"}
    if language:
        data["language"] = language
    files = {"file": (filename, audio_bytes, content_type or "audio/webm")}
    return await _request("POST", voice_url(), "/transcribe", "Voice (speech-to-text) module", data=data, files=files)
