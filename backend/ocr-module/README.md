# MediKiosk — Document OCR / Digitization

The "Scan" module from the architecture doc: upload a photo of a prior
prescription, lab report, or discharge summary, get back a structured,
confidence-flagged extraction, and a simple chronological timeline across
everything uploaded for one patient session.

**What this module does NOT do** — other owners' modules per the build plan:
- No conversational history — that's the separate `converse-module`.
- No FHIR/ABDM push. `GET /documents` and `GET /timeline` are what the
  backend/integration lead's module should consume to build a FHIR Bundle.
- **No drug-interaction checking.** The original problem statement mentions
  "potential drug interaction" flagging — that needs a real, licensed drug
  database to do responsibly, not something to fake with an LLM guess for a
  demo. Deliberately left out; say so if asked rather than pretending it's
  covered.
- No real kiosk UI. `static/index.html` is a bare test harness (pick a file,
  see the extraction) — not the patient-facing screen.

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

Same free API key as the Converse module — if you already set one up there,
reuse it here too (a fresh `aistudio.google.com` key or the same one both
work; the free-tier rate limit is per key, not per project). Otherwise:

1. https://aistudio.google.com/apikey → sign in → create a key. No card needed.
2. Copy `.env.example` to `.env`, paste it in as `GEMINI_API_KEY`.

**Privacy note — this one matters even more here than for the Converse
module:** you're uploading images of documents, not just typing symptoms.
Only ever use synthetic/sample documents (make one up, or find a blank
template online) during development and demos. Never photograph or upload
an actual prescription, lab report, or discharge summary — yours or anyone
else's — given the free tier's data-usage terms and the fact this module
has no consent flow.

Runs in **mock mode** with zero setup (no key needed) — same pattern as the
Converse module: a fixed, realistic-looking extraction keyed off whatever
`document_type_hint` you send, so the rest of the pipeline (upload, storage,
timeline, abnormal-value flagging) can be built and tested without burning
API calls or waiting on a key.

## Run it

```bash
uvicorn app.main:app --reload --port 8001
```

(Port 8001, not 8000 — so it can run alongside the Converse module if you
want both up at once.)

Open http://localhost:8001 for the test page: pick a file, optionally hint
the document type, hit Extract, and watch the structured JSON build up. The
timeline panel shows documents ordered by the date printed on them where
readable, with anything out-of-range highlighted.

Or hit the API directly:

```bash
# what's available right now
curl localhost:8001/backends

# upload and extract (document_type_hint and session_id are optional)
curl -X POST localhost:8001/documents \
  -F "file=@sample_prescription.jpg" \
  -F "document_type_hint=prescription" \
  -F "session_id=demo-1" \
  -F "backend=auto"

# everything uploaded for a session, in upload order
curl "localhost:8001/documents?session_id=demo-1"

# the same documents, chronologically where a date could be read
curl "localhost:8001/timeline?session_id=demo-1"
```

## Run the tests

```bash
pytest tests/ -v
```

Entirely against the mock backend with synthetic in-memory images — no
network, no key, no real document needed. Doesn't replace testing against
the real model with a few actual sample documents before a demo.

## Design decisions worth knowing about

- **Confidence is a first-class field, not an afterthought.** Every
  medication and investigation carries its own `confidence`, and the whole
  document carries `needs_review`/`review_reason`. The prompt (`app/prompts.py`)
  repeatedly tells the model not to guess — a fabricated dosage is a worse
  failure than an honest "couldn't read this."
- **`abnormal` is only ever set from a visible reference range on the
  document itself**, never from the model's general medical knowledge. If
  there's no reference range printed, it stays `null`. This was a deliberate
  call to avoid the module quietly making clinical judgments it has no
  business making.
- **The timeline never pretends false precision.** `document_date` is stored
  verbatim as read off the document; sorting is a best-effort parse
  (`dateutil`, day-first) that's allowed to fail. Undated documents sort by
  upload order and are listed after dated ones, each entry tagged with
  `date_source` so the UI is honest about which is which.
- **Same pluggable-backend shape as the Converse module.** `ocr_llm.py` and
  `mock_ocr.py` both expose `extract(image_bytes, mime_type,
  document_type_hint)`. If you want a local/offline fallback here too (the
  same argument that motivated Ollama for the Converse module applies), a
  vision-capable Ollama model (e.g. `llama3.2-vision`) would slot in the
  same way — not built yet, ask if you want it added.

## Known gaps — be upfront about these if asked

- Drug-interaction flagging: explicitly out of scope (see above).
- Handwriting extraction is best-effort and flagged, not solved — same
  caveat as everywhere else in this project.
- No auth, no persistence (in-memory, vanishes on restart), no dedup if the
  same document gets uploaded twice.
- 12MB upload cap and no batch upload — one document per request.

## Model name

Same as the Converse module: `GEMINI_MODEL` defaults to `gemini-3.5-flash`.
Check https://ai.google.dev/gemini-api/docs/models if it's been
deprecated/renamed by the time you're reading this.
