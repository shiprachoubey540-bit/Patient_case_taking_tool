"""
Local backend, using Ollama's structured-output support.

Verified against https://docs.ollama.com/capabilities/structured-outputs
(Sept 2026): POST /api/chat with a `format` field set to a JSON schema
(pydantic's `model_json_schema()` works directly) forces the response into
that shape.

Same run_turn(transcript, ayush_mode, turn_count) signature as llm.py and
mock_llm.py, so main.py can swap between them without caring which one it's
calling.

Context window (2026-09-07): previously no `num_ctx` was set, so Ollama
picked its own default — per https://docs.ollama.com/context-length that's
VRAM-heuristic-based (4k context under 24 GiB VRAM, more above that), not a
fixed number. That's a real risk here specifically because this module's
system prompt has grown three times over today (medications split,
triage_level, the Hindi language instruction) and the transcript itself
grows every turn up to session_store.MAX_TURNS=14 patient turns — a
worst-case interview (AYUSH mode + Hindi + a verbose patient) could plausibly
approach or exceed a 4k default, and a silently truncated system prompt is a
much worse failure mode than a slow response: it would drop instructions
(possibly including the red_flag/triage_level or "keep the record in
English" rules) without any error at all. OLLAMA_NUM_CTX now sets this
explicitly rather than trusting the heuristic to guess right on whatever
hardware this ends up running on.

keep_alive (2026-09-07): `ollama ps` on legion came back with an empty
table between interviews — meaning nothing was loaded in memory at that
moment. Ollama's own default keep_alive is 5 minutes
(https://github.com/ollama/ollama/blob/main/docs/api.md): the model unloads
that long after its last request and has to reload before the next one
answers. For local development that's a minor annoyance; for a kiosk where
real patients arrive with real gaps between them, it means the first
message of a fresh patient's interview routinely eats a cold-load on top of
normal inference time, and that cost was previously invisible — it wasn't
broken out from the rest of the turn's latency anywhere. OLLAMA_KEEP_ALIVE
now defaults to "30m" so an active clinic session keeps the model warm
end-to-end; run_turn also now logs `load_duration_s` (from Ollama's own
response, not a wall-clock guess) so a cold-load turn is visible and
distinguishable from genuinely slow inference, instead of the two being
conflated into one `elapsed=` number. Set OLLAMA_KEEP_ALIVE=-1 if this ends
up running on a machine dedicated to nothing else all day and you'd rather
just keep it loaded permanently — that trades held VRAM for zero reload
risk, a reasonable call for a real kiosk, less obviously right on a dev
laptop you use for other things too.

Cold-start warmup + heartbeat (2026-09-07): confirmed for real on legion —
the very first "headache" message after a fresh Ollama/converse-module
start took over 120s and hit run_turn's own httpx timeout, which fired
before Ollama replied at all (kiosk-backend's 220s outer timeout was NOT
the culprit here; ours was tighter and lost first). `ollama ps`, checked a
little later, showed the model warm and loaded on GPU — so Ollama did
eventually finish; the patient just wasn't there for it. The honest fix is
not to guess a bigger timeout number for a duration we don't actually know
(disk load, CUDA context setup, and — a real if unconfirmed suspect —
GBNF grammar compilation from this module's nested Pydantic schema could
all be contributing, and 120s might have needed to be 130s or 400s; we
don't know which). Instead, warmup() below sends a throwaway request at
server startup (main.py's lifespan) so the model is already loaded and any
one-time grammar compilation has already happened before a real patient
ever sends a message, and a periodic background heartbeat (also wired in
main.py) re-warms it well inside the keep_alive window so it never goes
cold again during operating hours. run_turn's own timeout is ALSO raised
(120s -> 200s) as a safety net for whatever this doesn't cover (e.g. Ollama
restarted mid-day, or warmup itself silently failed) — but the heartbeat,
not a bigger number, is the real fix; a bigger number only changes how long
a patient stares at "Still thinking..." before something (warmup or luck)
should have prevented them ever seeing a cold path at all.

CORRECTION (2026-09-07, same day): the warmup fix above shipped, and the
next log line it produced was `load_duration_s=0.0 ... elapsed=155.1s` —
model already warm (zero load time), and it STILL took two and a half
minutes for a single throwaway ping. That rules out cold model-loading as
the dominant cost; it was never really the bottleneck, or at best a minor
one.

SECOND CORRECTION (2026-09-07, same day again): with tok_per_s added to the
log, a real interview turn came back as `load_duration_s=0.0
prompt_tokens=2346 response_tokens=138 elapsed=49.1s` — generation itself
(eval_duration_s, response_tokens/eval_duration_s) was measured at ~41
tok/s, genuinely fine for a 12B model on GPU, and load was zero. But 138
response tokens at 41 tok/s is ~3.4s of actual generation — nowhere near
49.1s. Neither load nor generation explains the gap; PROMPT EVALUATION
(processing the ~2.3k input tokens before generation starts) is the only
remaining candidate, and it wasn't being measured at all — Ollama's
response includes `prompt_eval_duration` and this code just wasn't reading
it. If prompt eval is genuinely running at something like 50 tok/s here
instead of the hundreds-to-thousands/sec GPU prefill should manage, that
points at prompt processing being CPU-bound or otherwise not using the GPU
properly for the prefill step — a materially different, more actionable
problem than "the model is just slow," and not something guessable further
from what was logged before this fix. Both run_turn and warmup now log
prompt_eval_duration_s and a derived prompt_tok_per_s so the next real turn
settles this directly instead of leaving it as the last unmeasured gap.

Separately, worth flagging even though it's not a latency issue:
turn_runner.py's "empty turn" fallback (see that file's docstring) fired
TWICE IN A ROW on the very first real interview turn on legion — gemma4:12b
left next_question blank without setting complete=true, on both the
original attempt and the automatic retry, so the patient got the generic
"could you tell me more" fallback instead of a real follow-up question.
That's the known local-model quirk the retry/fallback logic exists for, but
seeing it trigger immediately (not as a rare edge case) is a real, separate
data point against trusting gemma4:12b's response quality as the default —
independent of whatever prompt-eval speed turns out to be. Worth watching
across more turns before concluding either way, but don't let a speed fix
alone stand in for checking this too.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import AsyncIterator

import httpx

from .prompts import build_system_instruction, render_transcript
from .schema import TurnResult

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b")
# See the context-window note above — 8192 is a deliberate margin over the
# worst-case estimate (~2.5-3k tokens for system prompt + a full 14-turn
# AYUSH+Hindi transcript), not a tuned/benchmarked number. If you have a
# specific VRAM budget to respect, set OLLAMA_NUM_CTX explicitly instead of
# trusting this default.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
# See the keep_alive note above. Ollama accepts either a duration string
# ("30m", "1h") or a bare number of seconds/(-1 for "never unload") — passed
# through as-is rather than parsed here, since Ollama already validates it.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
# Real-turn safety net, not the primary fix — see the cold-start note above.
# Must stay comfortably under kiosk-backend's own 220s proxy timeout
# (kiosk-backend/app/clients.py) or we recreate the exact timeout-mismatch
# bug that module's _TIMEOUT comment already warns about, just one layer
# further out.
_RUN_TURN_TIMEOUT_S = 200
# Warmup can legitimately take as long as a genuine cold start needs —
# nothing is waiting on it — so it gets a much longer leash than a real
# patient-facing turn does.
_WARMUP_TIMEOUT_S = 240


def _log_ollama_response(tag: str, body: dict, elapsed: float) -> None:
    """Shared by run_turn and warmup so their diagnostics can't drift apart
    again the way they did earlier today (warmup logged fewer fields than
    run_turn until this refactor). All duration fields come straight from
    Ollama's own response (nanoseconds -> seconds), not a wall-clock guess:
    load_duration (model load), prompt_eval_duration (processing the input
    before generation starts — the field that was missing entirely before
    today's second correction, above), and eval_duration (generation
    itself). elapsed - load - prompt_eval - eval should be small; if it
    isn't, something is happening that none of these three account for."""
    load_duration_s = body.get("load_duration", 0) / 1e9
    prompt_tokens = body.get("prompt_eval_count")
    prompt_eval_duration_s = body.get("prompt_eval_duration", 0) / 1e9
    prompt_tok_per_s = (prompt_tokens / prompt_eval_duration_s) if prompt_tokens and prompt_eval_duration_s else None
    response_tokens = body.get("eval_count")
    eval_duration_s = body.get("eval_duration", 0) / 1e9
    tok_per_s = (response_tokens / eval_duration_s) if response_tokens and eval_duration_s else None
    print(
        f"[llm/ollama] {tag} model={OLLAMA_MODEL!r} num_ctx={OLLAMA_NUM_CTX} keep_alive={OLLAMA_KEEP_ALIVE!r} "
        f"load_duration_s={load_duration_s:.1f} "
        f"prompt_tokens={prompt_tokens} prompt_eval_duration_s={prompt_eval_duration_s:.1f} "
        f"prompt_tok_per_s={f'{prompt_tok_per_s:.1f}' if prompt_tok_per_s else None} "
        f"response_tokens={response_tokens} eval_duration_s={eval_duration_s:.1f} "
        f"tok_per_s={f'{tok_per_s:.1f}' if tok_per_s else None} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )


def is_available(timeout: float = 1.5) -> bool:
    """Quick reachability check — used by /backends so the UI can grey out
    the option instead of letting a user hit a confusing error mid-demo."""
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def run_turn(
    transcript: list[dict],
    ayush_mode: bool = False,
    language: str = "en",
    turn_count: int = 1,
    retry_hint: str | None = None,
    current_history: str | None = None,
) -> TurnResult:
    if not is_available():
        raise RuntimeError(f"Ollama is not running or unreachable at {OLLAMA_URL}")

    t0 = time.monotonic()
    from . import rag
    clinical_guideline = rag.retrieve_guideline(transcript)
    system_instruction = build_system_instruction(
        ayush_mode, language, retry_hint, clinical_guideline=clinical_guideline, current_history=current_history
    )
    input_text = render_transcript(transcript)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": input_text},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "num_ctx": OLLAMA_NUM_CTX},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    started = time.monotonic()
    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=_RUN_TURN_TIMEOUT_S)
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` running, "
            f"and has `{OLLAMA_MODEL}` been pulled? (`ollama pull {OLLAMA_MODEL}`)"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned an error: {exc.response.text}") from exc
    elapsed = time.monotonic() - started

    body = resp.json()
    content = body["message"]["content"]
    _log_ollama_response(f"turn_count={turn_count}", body, elapsed)
    # from_raw_lenient (not the plain model_validate_json) — see its
    # docstring in schema.py: needed since next_question's min_length=1 fix
    # (2026-09-12) means a backend that still emits an empty string despite
    # the schema now fails validation for the WHOLE object, which would
    # otherwise throw away a perfectly good history re-derivation over one
    # bad field and surface as a raw 502 instead of turn_runner's normal
    # retry/fallback handling.
    return TurnResult.from_raw_lenient(content)


async def astream_turn(
    transcript: list[dict],
    ayush_mode: bool = False,
    language: str = "en",
    turn_count: int = 1,
    current_history: str | None = None,
) -> AsyncIterator[str]:
    """Async generator version of run_turn for the "live tokens" streaming
    endpoint (main.py's POST .../message/stream, 2026-09-12) — yields raw
    content chunks exactly as Ollama produces them (stream=True), for the
    caller to feed into stream_extract.IncrementalFieldExtractor so the
    patient sees acknowledgement/next_question appear progressively instead
    of all at once.

    Deliberately does NOT take a retry_hint and does NOT do turn_runner.py's
    empty-turn retry itself — this is only ever the FIRST attempt. The
    caller accumulates the full text as it streams, and if the final result
    turns out empty/invalid (rare now that schema.py enforces
    min_length=1 — see that field's docstring), falls back to a plain
    (non-streamed) run_turn(..., retry_hint=RETRY_HINT) call and tells the
    frontend to replace whatever partial text was already shown. Streaming
    the retry too would double this whole file's complexity for a path
    that should now be uncommon — not worth it for a demo.

    Same warnings apply as run_turn: a ConnectError/HTTPStatusError becomes
    a plain RuntimeError so main.py can surface a clean error event instead
    of an unhandled exception killing the stream mid-response.
    """
    from . import rag
    clinical_guideline = rag.retrieve_guideline(transcript)
    system_instruction = build_system_instruction(
        ayush_mode, language, clinical_guideline=clinical_guideline, current_history=current_history
    )
    input_text = render_transcript(transcript)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": input_text},
        ],
        "stream": True,
        "format": "json",
        "options": {"temperature": 0.3, "num_ctx": OLLAMA_NUM_CTX},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    started = time.monotonic()
    final_body: dict = {}
    try:
        async with httpx.AsyncClient(timeout=_RUN_TURN_TIMEOUT_S) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    error_bytes = await resp.aread()
                    raise RuntimeError(f"Ollama returned an error: {error_bytes.decode(errors='replace')}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    content = obj.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if obj.get("done"):
                        final_body = obj
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` running, "
            f"and has `{OLLAMA_MODEL}` been pulled? (`ollama pull {OLLAMA_MODEL}`)"
        ) from exc

    elapsed = time.monotonic() - started
    if final_body:
        _log_ollama_response(f"turn_count={turn_count} STREAM", final_body, elapsed)


def warmup() -> bool:
    """Load the model (and let Ollama do whatever one-time setup a fresh
    load involves — disk read, VRAM placement, possibly grammar compilation
    for this module's schema) with nobody waiting on it, instead of a real
    patient's first message paying that cost. Called from main.py's startup
    lifespan and again periodically from a background heartbeat — see the
    cold-start note at the top of this file for why both exist. Deliberately
    swallows every exception and just reports success/failure: a warmup
    failure (Ollama not running yet, still starting up itself, momentarily
    unreachable) must never crash converse-module or block it from serving
    other backends — a plain run_turn() call still gets its own honest error
    the next time a patient actually tries the ollama backend.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            # Minimal but real content, not empty — an empty user message
            # risks Ollama short-circuiting before doing the same schema/
            # grammar setup a real turn triggers, which would defeat the
            # point of warming up in the first place.
            {"role": "system", "content": build_system_instruction(False, "en")},
            {"role": "user", "content": "(warmup ping — not a real patient turn)"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "num_ctx": OLLAMA_NUM_CTX},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    started = time.monotonic()
    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=_WARMUP_TIMEOUT_S)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[llm/ollama] warmup failed (will retry on next heartbeat): {exc!r}", file=sys.stderr)
        return False
    elapsed = time.monotonic() - started
    body = resp.json()
    _log_ollama_response("warmup OK", body, elapsed)
    return True
