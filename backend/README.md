# MediKiosk

A four-service prototype for a patient self check-in kiosk (SIH26047):
conversational intake, document OCR, voice-to-text, and the kiosk screen
that ties them together, plus a staff-facing doctor view. Each service has
its own README with the actual detail — this file is just "how do I get
the whole thing running":

- [`kiosk-backend/README.md`](kiosk-backend/README.md) — the patient-facing
  screen and the staff/doctor view; the one you actually open in a browser
- [`converse-module/README.md`](converse-module/README.md) — the
  conversational intake engine (Gemini / Ollama / mock)
- [`ocr-module/README.md`](ocr-module/README.md) — document photo scanning
- [`voice-module/README.md`](voice-module/README.md) — speech-to-text for
  the kiosk's mic button

## Setup — one shared venv (added 2026-09-11)

All four services' dependencies now live in one `requirements.txt` at the
repo root, so there's one venv to create instead of four:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # voice-module's faster-whisper/ctranslate2 make this the slow part — give it a few minutes
```

That's it for dependencies — `run_all.py` (below) looks for this venv first.

Each module still needs its own `.env` if you want the real Gemini/Ollama
backends instead of mock mode — see each module's own README for that
(API keys, Ollama setup, etc.). None of that is affected by sharing one venv;
it's purely a dependency-install convenience, not a merge of the services
themselves. They're still four independent processes on four ports — see
kiosk-backend/README.md's "Why separate services instead of merging into
one?" section for why that's deliberate, not something this change reversed.

**If you'd rather keep each module's dependencies fully isolated** (e.g. you
only want to run one service on a machine that will never run the others),
each module's own `requirements.txt` still works exactly as before — just
`cd` into that module and make its own `.venv` there instead of at the repo
root. `run_all.py` falls back to a per-module `.venv/` automatically if the
shared one at the root doesn't exist, so nothing from before 2026-09-11
breaks.

## Run everything

```bash
python run_all.py
```

Starts all four services, streams their logs into one terminal with
`[name]` prefixes, and stops all four cleanly on one Ctrl+C. Then open:

- patient check-in: http://localhost:8002
- staff/doctor view: http://localhost:8002/doctor.html (no login — see that
  page's own on-page warning before putting it anywhere but a trusted local
  network)

See `kiosk-backend/README.md` for what the actual check-in flow does, and
its "Known gaps" section for an honest list of what this prototype does not
yet do.
