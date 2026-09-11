"""
Module B — Medical Document Digitization & Intelligence
API service. Other modules (A, C, D) call POST /digitize with a
patient-uploaded file (image or PDF) and get back structured JSON.

Run locally:
    uvicorn app.main:app --reload --port 8001
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from app.preprocessing import preprocess_image, pdf_bytes_to_images
from app.ocr_engine import ocr_pages
from app.entity_extractor import extract_entities

app = FastAPI(title="MediKiosk - Module B: Document Digitization")


@app.get("/health")
def health():
    return {"status": "ok", "module": "B - Document Digitization"}


@app.post("/digitize")
async def digitize_document(file: UploadFile = File(...)):
    """
    Accepts: image (jpg/png) or PDF of a prescription/lab report/discharge
    summary.

    Returns structured JSON that Module C consumes to merge with the
    conversational history from Module A:
        {
          "source_filename": ...,
          "ocr_confidence": 0-100,
          "needs_human_review": bool,
          "raw_text": "...",
          "structured_data": {
              "medicines": [...],
              "diagnoses": [...],
              "dates": [...],
              "lab_values": [...]
          }
        }
    """
    content = await file.read()
    filename = file.filename or "unknown"

    try:
        if filename.lower().endswith(".pdf"):
            pil_pages = pdf_bytes_to_images(content)
        else:
            pil_pages = [Image.open(io.BytesIO(content))]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    processed_pages = [preprocess_image(p) for p in pil_pages]
    ocr_result = ocr_pages(processed_pages)
    structured = extract_entities(ocr_result["text"])

    return {
        "source_filename": filename,
        "ocr_confidence": ocr_result["avg_confidence"],
        "needs_human_review": ocr_result["needs_review"],
        "raw_text": ocr_result["text"],
        "structured_data": structured,
    }
