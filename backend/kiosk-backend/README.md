# MediKiosk — Patient Kiosk

The actual patient-facing screen from the architecture doc, tying the other
three modules together into one check-in flow: welcome → conversational
intake (typed or **spoken**) → optional document upload → confirmation →
done.

**This module has no clinical logic of its own.** It's a thin orchestration
layer that calls `converse-module` (the interview), `ocr-module` (the
document scan), and `voice-module` (speech-to-text for the mic button) over
HTTP and merges their results for the UI. If you're looking for the intake
questions, the extraction rules, or the transcription logic, they live in
those three modules, not here.

## Why separate services instead of merging into one?

Because that's how the other modules were actually built and owned — as
independent, separately runnable/testable services with their own ports and
their own backend toggles. A kiosk needs one patient visit to span all of
them, but that's an orchestration problem, not a reason to merge everything
into one codebase. If you'd rather have one process, the honest alternative
is a rewrite, not a shortcut — flagging this because "just combine them"
seems simpler than it is once you actually look at how different their
session models are (Converse's `session_id` vs OCR's `session_id`-as-
grouping-key aren't the same kind of thing, and voice-module has no session
concept at all — it's stateless, just audio in, text out).

## Setup

You need **all four services** running for the kiosk to actually work —
this one alone will start fine but every screen past "Welcome" will show a
"couldn't reach the service" error until the others are up, and the mic
button hides itself if voice-module specifically isn't reachable.

**As of 2026-09-11, first-time setup is ONE shared venv for all four
services** — see the [top-level README](../README.md)'s Setup section
(one level up from this module) and run that once; it covers this module
too. That's the recommended path now.

If you'd rather keep each module's dependencies fully isolated (its own
venv per module, nothing shared) that still works exactly as before:

```bash
# repeat for converse-module, ocr-module, voice-module, kiosk-backend
cd <module-dir>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # voice-module's is heavier — pulls in faster-whisper/ctranslate2
```

**After that, one command starts all four at once** — `run_all.py`, at the
repo root (next to this module's folder, not inside it):

```bash
python run_all.py
```

It starts each service in its own venv (auto-detected per module), streams
all four logs into one terminal with `[name]` prefixes so you can tell them
apart, and one Ctrl+C shuts all four down cleanly. It's a convenience
wrapper only — nothing about how the services run is different, it just
saves you four terminal windows. If you'd rather see one service's log in
isolation (e.g. debugging just voice-module without the others' noise), the
original four-terminal way still works exactly as before:

```bash
# terminal 1 — the intake engine
cd converse-module && source .venv/bin/activate && uvicorn app.main:app --port 8000

# terminal 2 — the document scanner
cd ocr-module && source .venv/bin/activate && uvicorn app.main:app --port 8001

# terminal 3 — speech-to-text for the mic button
cd voice-module && source .venv/bin/activate && uvicorn app.main:app --port 8003

# terminal 4 — this module
cd kiosk-backend && source .venv/bin/activate && uvicorn app.main:app --port 8002
```

Either way, open http://localhost:8002 — that's the actual kiosk screen (not
a bare test harness like the other modules' `static/index.html`).

No API key needed for this module itself — it doesn't call any LLM
directly. All three other modules run in mock mode with zero setup, so the
whole four-service flow works offline out of the box; add `GEMINI_API_KEY`
to converse/ocr's own `.env` files (see their READMEs) when you want the
real thing.

**First time you tap the mic**, voice-module downloads the Whisper model
weights from Hugging Face (a few hundred MB for the default "small" model)
— that needs internet access and can take a minute or two. After that first
download it's cached locally and runs fully offline. If that download fails
partway (flaky wifi, a firewall blocking huggingface.co) the mic will show
"Couldn't transcribe that — please type your answer instead" rather than
silently retrying — the patient can always just type, so this degrades
gracefully, but worth knowing about before a live demo: consider doing one
throwaway test recording beforehand on the actual demo machine/network so
the download already happened.

If any service is on a different port or machine, override `CONVERSE_URL` /
`OCR_URL` / `VOICE_URL` — copy `.env.example` to `.env` and edit.

## Run the tests

```bash
pytest tests/ -v
```

These do **not** need the other services running — `respx` intercepts
the outbound HTTP calls this module makes and returns canned responses, so
the tests check this module's own orchestration logic (routing, error
handling, the ollama→auto fallback for document scans, summary merging)
in isolation. They don't prove the real Converse/OCR/Voice modules behave
the way the mocks assume — that's what the manual four-terminal run above
is for.

## What the flow actually does

1. **Welcome** — patient types their name (not verified — just a display
   label) and optionally flags themselves as an AYUSH patient. A collapsed
   "Kiosk settings" section lets staff pick an intake engine (auto / gemini
   / ollama / mock) for testing; patients aren't expected to touch it.
2. **Intake** — a chat UI against the Converse module. Answers can be typed
   or spoken: tapping the mic records audio, sends it to voice-module for
   transcription, and fills the text box with the result for the patient
   (or a staff member) to review and edit — it's never auto-submitted. The
   mic button hides itself entirely if voice-module isn't reachable, and a
   staff-only badge on the welcome screen says "mock only (no Whisper)" if
   the service is up but the real model isn't available, so nobody's
   surprised by canned transcripts during a demo. If a red-flag keyword
   is detected, the flow jumps straight to an urgent-attention screen and
   stops — no further questions, no summary, just "tell the staff now."
3. **Documents** — optional photo upload(s) against the OCR module. Can be
   skipped entirely.
4. **Summary** — pulls `GET /visits/{id}/summary` (this module's own
   endpoint, which itself calls both other services) and renders it as
   readable sections, not raw JSON — this is the one screen in the whole
   project meant for a patient to actually read, so it's deliberately not
   styled like the other two modules' dev-facing JSON dumps.
5. **Done** — shows the patient a 6-character visit code (e.g. `A7K2M9`) —
   see "Persistence &amp; the doctor view" below for what it's for and what
   it isn't. "Start next patient" resets the kiosk for reuse without a page
   reload, since a physical kiosk will be used back to back all day.

These five screens are each their own file now (`templates/screens/*.html`,
reorganized 2026-09-11) instead of one long HTML file — purely for
readability. `templates/index.html` still assembles all five into one page
via Jinja `{% include %}`s and serves it from `GET /`, and the JS still
does client-side show/hide between them with no page reload — nothing
about how the patient experiences the flow changed, only how the markup is
organized on disk. `templates/doctor.html` is served the same way from
`GET /doctor.html`. Both routes live in `app/main.py`; there's no separate
`static/` folder any more.

## Persistence & the doctor view (added 2026-09-11)

Two things that used to not exist at all:

- **A visit now survives a kiosk-backend restart.** Every check-in, message,
  and document upload writes a snapshot to a local SQLite file
  (`kiosk-backend/data/medikiosk.db` by default — override with
  `MEDIKIOSK_DB_PATH` in `.env`). On startup, the server reloads every visit
  from that file back into memory so an in-progress visit keeps working.
  **This does NOT cover converse-module or ocr-module restarting** — those
  two still keep their own session state only in their own memory, so a
  restart of either of THEM mid-interview still breaks that patient's live
  conversation exactly as before. What's different is that whatever was
  captured up to that point is still visible afterward instead of vanishing
  outright. See `app/db.py`'s module docstring for the full scope — read it
  before assuming more is covered than actually is.
- **`GET /doctor.html`** — a staff-facing page: type in a patient's 6-char
  visit code to see their full recorded intake, or just watch the queue
  (sorted red-flag/urgent first, then most-recently-updated). **There is no
  login on this page** — same "not faked for a demo" stance as patient
  identity below. It's fine for a kiosk on a trusted local network during a
  pilot; do not put this on any network a patient or the public can reach
  without adding real staff authentication first. The page says this too,
  loudly, so it isn't missed by someone who only reads code.

The visit code is a kiosk-generated lookup key, **not a verified patient
ID** — same honesty caveat as the free-text name field below, one level up.
There's no ABHA/identity-registry lookup behind it; it just lets staff find
a specific visit again without an actual patient-ID system existing yet.

## Known gaps — be upfront about these if asked

- **No calling/waiting-room integration.** "Please take a seat, you'll be
  called shortly" is still just UI copy — the 6-character code from
  "Persistence & the doctor view" above is a real lookup key now, but there's
  no connection to an actual waiting-room display, PA announcement, or
  calling system. The doctor queue page is the only place it's surfaced.
- **No patient identity verification.** The name field is free text. A real
  deployment needs ABHA/patient-ID lookup here; deliberately not faked — and
  the new visit code is a kiosk-generated lookup token, not a substitute for
  that (see the caveat above).
- **kiosk-backend's persistence is real but narrow.** A visit survives THIS
  module restarting (see above) — that's new as of 2026-09-11 and is a
  genuine change from before. converse-module and ocr-module are unchanged:
  they're still in-memory only, so either of them restarting mid-interview
  still breaks that patient's live conversation. No dedup if a patient
  accidentally starts two visits.
- **The Ollama intake backend has no OCR equivalent.** If a visit picks
  "ollama" as its engine, document scans for that visit silently use "auto"
  instead (falls back to Gemini if a key is set, else mock) — the check-in
  response says so via `ocr_backend_note`, and the UI surfaces it in the
  header, but it's easy to miss. Said here so it isn't a surprise.
- **No retry/offline queueing** if a service drops mid-visit. An error
  banner shows and the patient can retry the same action, but a half-typed
  answer isn't auto-saved anywhere beyond the current in-memory visit.
- **"Whisper available" only means the library imported successfully**,
  not that a real transcription will actually succeed — `/backends`
  reports whisper as available as soon as `faster-whisper` is importable,
  before it has ever tried to load model weights. This module always asks
  voice-module for its "auto" backend when transcribing (not whatever
  engine the visit itself is using), so if the model download fails at the
  moment someone taps the mic, they'll see a real error ("Couldn't
  transcribe that") even though the welcome screen said whisper was fine.
  Verified this exact failure mode end-to-end during testing — it degrades
  safely (patient types instead), but it's worth knowing it can happen and
  isn't a bug if it does.
- This is a prototype for a hackathon pitch, not a validated clinical
  workflow — the footer disclaimer on the welcome screen says this for a
  reason.

## Model name

This module makes no LLM calls itself, so there's nothing to check here —
see the other modules' READMEs for their model-name caveats (converse-module
and ocr-module for Gemini/Ollama, voice-module for the Whisper model size).
