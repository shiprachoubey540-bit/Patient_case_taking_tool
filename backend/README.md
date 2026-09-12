# MediKiosk — Conversational History Engine

The "Converse" module from the architecture doc: a FastAPI service that runs the
patient interview (chief complaint → SOCRATES-guided HPI → past/drug/family
history → optional AYUSH intake) and returns a structured `ClinicalHistory`
JSON, one field at a time, over a simple HTTP API.

**What this module does NOT do** — these are other owners' modules per the
build plan, and this module is deliberately built to hand off to them cleanly:
- No speech-to-text/text-to-speech. Feed it text — however that text was
  produced (typed, or transcribed client-side/by a separate Bhashini-backed
  service). It never touches audio.
- No document OCR.
- No FHIR/ABDM push. `GET /sessions/{id}/history` returns the structured JSON
  the backend/integration lead's module should consume to build a FHIR Bundle.
- No real kiosk UI. `static/index.html` is a bare test harness to try the
  interview loop and watch the JSON build up — not the patient-facing screen.

## Setup

**As of 2026-09-11, the whole project shares one venv** — see the
[top-level README](../README.md)'s Setup section and just run that once;
it covers this module too. That's the recommended path.

Only if you want to run *this module alone*, isolated from the other
three (its own venv, nothing shared) — still works exactly as before:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get a free API key

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Create a key — free tier, **no credit card required**.
3. Copy `.env.example` to `.env` and paste it in as `GEMINI_API_KEY`.

**Privacy note, worth taking seriously for a healthcare pitch:** on the free
tier, Google's terms allow prompts/responses to be used to improve their
products (paid tiers turn this off). Only ever put synthetic/dummy patient
data through this during development and demos — never anything resembling
a real patient's information. That's true independent of the API terms too:
this module has no consent flow yet, so nothing here is DPDP-ready.

Free tier is rate-limited (roughly 15 requests/minute, 250/day per key as of
this writing) — fine for development and a live demo, tight if the whole
team shares one key while testing. Everyone should grab their own free key.

Each teammate can run in **mock mode** with zero setup — just don't set
`GEMINI_API_KEY` and the server automatically falls back to a scripted,
no-network flow (see `app/mock_llm.py`). Useful for the other five people on
the team to build against this module's API before they have a key, and as
an offline fallback if the real API is flaky right before a demo.

### Optional: a local model via Ollama

There's a third backend, `ollama`, that talks to a locally-running
[Ollama](https://ollama.com) instead of the cloud. It sidesteps the
free-tier privacy caveat above entirely (nothing leaves the machine — no
"Google's terms allow using this to improve their products" question at
all), and it means no dependency on the clinic having reliable internet.

**Ollama is the intended default (decided 2026-09-07),** not a demo-day
fallback — an earlier version of this paragraph said otherwise, but that had
already fallen out of sync with the code: `resolve_backend()` in
`app/main.py` makes `"auto"` (the kiosk UI's default selection) prefer
Ollama over Gemini whenever it's reachable, and has for a while — see that
function's docstring and `test_auto_prefers_ollama_over_gemini_when_both_available`
in `tests/test_flow_mock.py`. Privacy and offline reliability are the actual
point, not a stage contingency. What's NOT yet verified is that a local
model's JSON-schema-following and clinical judgment (`red_flag`,
`triage_level`) hold up as well as Gemini's do when it's the primary path
every patient goes through, not an occasional backup — that needs real
interview-by-interview testing (English and Hindi, routine and red-flag
cases), not just a doc change. Treat that as open until someone's actually
watched it happen.

```bash
ollama serve                     # if it isn't already running
ollama pull gemma4:12b           # or whichever model you have — see OLLAMA_MODEL below
```

It's picked up automatically — no `.env` change needed unless you're using a
different model, a non-default Ollama port, or want to tune the context
window:

```
OLLAMA_MODEL=gemma4:12b
OLLAMA_URL=http://localhost:11434
OLLAMA_NUM_CTX=8192
OLLAMA_KEEP_ALIVE=30m
OLLAMA_HEARTBEAT_MINUTES=20
```

**GPU confirmed (2026-09-07):** `ollama ps` on legion, checked while a turn
was actually loaded, showed `100% GPU` — not CPU-bound, so that's ruled out
as a concern on this machine. `CONTEXT 8192` and a live `keep_alive`
countdown in that same output confirmed `OLLAMA_NUM_CTX`/`OLLAMA_KEEP_ALIVE`
are actually being applied, not just set in `.env` and ignored.

**Cold-start, hit for real (2026-09-07):** the very first patient message
after a fresh start of Ollama/converse-module took over 120s and errored
out as "history engine error: timed out" — converse-module's own timeout to
Ollama fired before Ollama had actually replied (not kiosk-backend's outer
220s; that one wasn't even reached). `ollama ps` checked a bit later showed
the model loaded fine — so Ollama did eventually finish, the patient just
wasn't there to see it. The real fix isn't a bigger timeout number for a
cold-start duration nobody's actually measured (disk load, GPU/VRAM setup,
and possibly one-time grammar compilation from this module's schema could
all be contributing — unconfirmed which). Instead:
- `app/main.py`'s startup now fires a **warmup request** in the background
  the moment the server boots, so the model is already loaded before any
  real patient can reach it (uvicorn starts serving immediately either way —
  this doesn't delay startup).
- A **background heartbeat** (`OLLAMA_HEARTBEAT_MINUTES`, default 20 —
  intentionally under the 30-minute `OLLAMA_KEEP_ALIVE`) keeps re-warming it
  for as long as the server runs, so a slow afternoon with big gaps between
  patients can't let it go cold again either.
- `run_turn`'s own timeout to Ollama is still raised (120s → 200s) as a
  safety net for whatever the above doesn't catch — but treat that as a
  backstop, not the fix; if a patient is hitting it, warmup/heartbeat should
  be checked first (`[llm/ollama] warmup OK ...` should appear in the log
  at startup and every ~20 minutes after).

Every Ollama turn (including warmups) logs to stderr —
`[llm/ollama] turn_count=... keep_alive=... prompt_tokens=... response_tokens=... load_duration_s=... elapsed=...s`
— `load_duration_s` comes straight from Ollama's own response and tells you
directly whether a slow turn was a cold reload (high `load_duration_s`) or
genuinely slow inference (`load_duration_s` near zero), and `prompt_tokens`
tells you whether `OLLAMA_NUM_CTX` is actually being approached.

## Run it

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 for the test chat page — there's a **Model**
dropdown (Auto / Gemini / Local — Ollama / Mock) that lets you pick the
backend per session; options grey themselves out automatically if that
backend isn't currently reachable (checked live via `GET /backends`).

Or hit the API directly:

```bash
# check what's available right now
curl localhost:8000/backends

# start a session on a specific backend ("auto" | "gemini" | "ollama" | "mock")
curl -X POST localhost:8000/sessions -H 'Content-Type: application/json' \
  -d '{"ayush_mode": false, "backend": "auto"}'

# send a patient turn
curl -X POST localhost:8000/sessions/<session_id>/message \
  -H 'Content-Type: application/json' -d '{"text": "My stomach hurts"}'

# read the structured history at any point
curl localhost:8000/sessions/<session_id>/history
```

A session's backend is fixed at creation and validated up front — asking for
`"gemini"` with no key configured, or `"ollama"` with nothing listening on
`OLLAMA_URL`, gets a clear `400` immediately instead of failing confusingly
mid-interview. `"auto"` (the default) prefers **Ollama** if it's reachable,
then Gemini if a key is configured, otherwise mock.

Local is preferred in "auto" on purpose: no patient symptom text leaves the
machine over a per-turn call to a third-party API, and the kiosk keeps
working if the clinic's internet is down or flaky — both matter more for a
real OPD deployment than shaving off Gemini's ~10-20s-per-turn latency,
though that's a real win too. The tradeoff: `llama3.1:8b` is a general
model with no medical fine-tuning, and it's already shown it's less
reliable than Gemini at following the structured-output schema exactly
(see `turn_runner.py`'s fail-soft fallback, added after it happened) — Local
being preferred by default doesn't mean it's proven more dependable, just
that latency and privacy were judged to matter more for this use case.
Override with an explicit `"backend"` choice if you want to force Gemini or
mock instead.

## Run the tests

```bash
pytest tests/ -v
```

These run entirely against mock mode — no network, no API key needed — and
are the fast way to check nothing broke after a change. They don't replace
testing against the real model before a demo; run through the actual
interview at least once with a real key beforehand.

## Design decisions worth knowing about

- **Full-state re-derivation, not patches.** Each turn, the model is asked
  to output the *entire* current `ClinicalHistory` based on the whole
  transcript so far, not just a delta. Costs a bit more context per call,
  but avoids an entire class of merge bugs — worth it at the scale of one
  patient conversation.
- **The red-flag check is rule-based first, LLM second.** `app/safety.py`
  runs a keyword check on every patient message *before* the LLM call.
  Emergency detection shouldn't depend on a model call succeeding — this is
  defense in depth, not a replacement for the LLM's own judgment (which
  still runs and can also set `red_flag`).
- **A hard turn cap exists independent of the model's judgment.**
  `session_store.MAX_TURNS` (currently 14 patient turns) stops the interview
  even if the model never sets `complete=true`. Don't rely on the model to
  self-regulate length live on stage.
- **Sessions are in-memory and vanish on restart.** Fine for a hackathon
  demo. If this needs to survive restarts later, swap `session_store.py` for
  a real store — nothing else in the codebase should need to change.
- **Three backends, one interface.** `llm.py`, `ollama_llm.py`, and
  `mock_llm.py` all expose the same `run_turn(transcript, ayush_mode,
  turn_count)` signature and return a `TurnResult`. A session picks one at
  creation time (`main.py`'s `resolve_backend`/dispatch) and never switches
  mid-conversation. Adding a fourth backend later means writing one more
  file with that signature, not touching the request-handling code.

## Known gaps — be upfront about these if asked

- **The AYUSH questions (`app/prompts.py`, `AyushAssessment` in
  `app/schema.py`) are a structural placeholder**, not a clinically reviewed
  Dashavidha Pariksha assessment. Get an Ayurvedic practitioner to look at
  the actual questions before presenting this as more than a first draft —
  this is the one claim in the whole pitch that can't be faked under
  questioning from an AIIA-affiliated panel.
- The red-flag keyword list is a starting point, not a validated one.
- No auth, no persistence, no consent flow — all explicitly out of scope for
  this module, but don't let the demo imply otherwise.

## Model name

`GEMINI_MODEL` defaults to `gemini-3.5-flash` (current free-tier flash model
as of September 2026, per https://ai.google.dev/gemini-api/docs/models).
Google renames/deprecates these — if calls start failing with a model-not-
found error, check that page and update `.env`.

## Gemini latency (thinking level)

If Gemini-backend turns feel slow (30-60s+ by mid-interview), check
`GEMINI_THINKING_LEVEL` in `.env` first before assuming something else is
wrong. `app/llm.py` now explicitly sets `generation_config.thinking_level`
on every call — previously it set nothing, and `gemini-3.5-flash` defaults
to `thinking_level="medium"` in that case, which is a real extended-
reasoning pass, not free. That was confirmed as the cause of a live report
of 1-minute+ turns on legion (2026-09-07): every turn re-derives the whole
structured history from the full transcript already (see the design note in
`app/prompts.py`), so a "medium" reasoning pass on top of a growing
transcript compounds badly as the interview goes on.

Default is now `low`. If it's still too slow, try `minimal` in `.env` — but
watch `triage_level`/`red_flag` quality closely first; `minimal` is
documented as tuned for quick factual answers, not judgment calls, and
those two fields are the one place this module still asks the model to
weigh something rather than just transcribe it. The keyword safety net in
`app/safety.py` catches obvious emergencies regardless of what the LLM
decides, but it's a backstop, not a reason to stop caring about
`triage_level` quality.

Every Gemini turn now also logs `elapsed=` to stderr
(`[llm/gemini] turn_count=... elapsed=...s`) — use that instead of timing
turns by eye from the UI if latency needs checking again later.
