# Module B — Medical Document Digitization & Intelligence

Part of MediKiosk (SIH26047, "Patient Case-Taking Software").

## What it does
Takes a patient-uploaded document (photo or PDF of a prescription, lab
report, or discharge summary), cleans it up, OCRs it, and pulls out
structured clinical data — medicines, dosages, diagnoses, dates, lab
values — as JSON.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
You also need Tesseract OCR and Poppler (for PDF->image) installed
system-wide, not just the Python wrappers:
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki (Windows installer)
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases
  (add the `bin` folder to PATH)

## Run
```bash
uvicorn app.main:app --reload --port 8001
```
Test at http://localhost:8001/docs (FastAPI auto-generates a Swagger UI
where you can upload a test file directly in the browser — good for
your demo too).

## Pipeline
1. **preprocessing.py** — deskew + denoise + binarize the image so OCR
   isn't fighting a blurry phone photo of a prescription.
2. **ocr_engine.py** — Tesseract extraction, with per-word confidence so
   low-confidence reads (e.g. an ambiguous dosage number) get flagged
   for human review instead of silently trusted.
3. **entity_extractor.py** — regex/heuristic extraction of medicines,
   dosage, frequency, diagnoses, dates, lab values from the raw OCR
   text. This is the piece to upgrade first if you have time —
   swap in a trained NER model (spaCy custom model or scispaCy) once
   you have sample documents to build a proper entity list from.
4. **main.py** — FastAPI wrapper exposing `POST /digitize`.

## How this connects to the other modules
- **Module A** (conversational history engine) captures the patient's
  spoken/typed history. It doesn't call Module B directly — they're
  parallel inputs.
- **Module C** (structured history summary generator) is the one that
  calls your `/digitize` endpoint for each uploaded document, then
  merges the returned `structured_data` with Module A's conversational
  output into one case summary. Agree the merge key (e.g. patient
  session ID) with whoever owns Module C.
- **Module D** (consent/privacy/ABDM) sits around everything — it should
  gate document upload behind consent capture before your endpoint is
  even called, and handles how the final data is stored/shared under
  ABDM. You don't need to implement consent logic yourself, just make
  sure your endpoint doesn't process anything before consent is
  confirmed upstream.

## What to bring to your team sync
- Exact document types you'll be tested on in the demo (prescriptions
  only, or also lab reports/discharge summaries — changes what regex
  patterns/vocab you need).
- The JSON schema Module C actually expects, so field names match.
- Whether Module D needs you to encrypt/delete the raw image after
  processing (likely yes for a health-data judge criterion).
