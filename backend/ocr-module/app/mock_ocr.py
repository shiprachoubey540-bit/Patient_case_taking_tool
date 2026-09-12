"""
No-network stand-in for ocr_llm.py — same purpose as the Converse module's
mock_llm.py: build/test the rest of the pipeline (upload, storage, timeline,
abnormal flagging) without needing a key or a network call, and as an
offline fallback.

It doesn't actually look at the image content — it returns a fixed,
realistic-looking extraction keyed off document_type_hint, so tests can
exercise both a "clean printed document" case and a "flagged for review"
case deterministically.
"""
from __future__ import annotations

from .schema import ExtractedDocument, Investigation, Medication, Vitals


def extract(image_bytes: bytes, mime_type: str, document_type_hint: str | None = None) -> ExtractedDocument:
    if document_type_hint == "lab_report":
        return ExtractedDocument(
            document_type="lab_report",
            document_date="04/09/2026",
            diagnoses=[],
            medications=[],
            investigations=[
                Investigation(
                    name="Hemoglobin", value="10.2", unit="g/dL", reference_range="12.0-15.5", abnormal=True, confidence=0.95
                ),
                Investigation(
                    name="Fasting Blood Glucose", value="92", unit="mg/dL", reference_range="70-100", abnormal=False, confidence=0.97
                ),
            ],
            procedures_surgeries=[],
            raw_text_excerpt="[mock] Hemoglobin 10.2 g/dL (12.0-15.5) ... Fasting Blood Glucose 92 mg/dL (70-100)",
            overall_confidence=0.96,
            needs_review=False,
        )

    if document_type_hint == "prescription":
        return ExtractedDocument(
            document_type="prescription",
            document_date="02/09/2026",
            presenting_complaints=["Giddiness", "Restlessness"],
            diagnoses=["Suspected viral fever"],
            vitals=Vitals(blood_pressure="110/70", pulse_rate="60bpm"),
            medications=[
                Medication(name="Paracetamol", dosage="650mg", frequency="twice daily", confidence=0.4),
                Medication(name="(illegible)", dosage=None, frequency=None, confidence=0.15),
            ],
            investigations=[],
            procedures_surgeries=[],
            raw_text_excerpt="[mock] c/o giddiness, restlessness ... O/E BP 110/70, PR 60bpm ... "
            "Paracetamol 650mg BD x 3 days ... [illegible second line]",
            overall_confidence=0.35,
            needs_review=True,
            review_reason="Handwritten prescription, second medication line illegible",
        )

    return ExtractedDocument(
        document_type="other",
        document_date=None,
        diagnoses=[],
        medications=[],
        investigations=[],
        procedures_surgeries=[],
        raw_text_excerpt="[mock] no document_type_hint given — generic placeholder extraction",
        overall_confidence=0.5,
        needs_review=True,
        review_reason="Mock backend cannot classify without a document_type_hint",
    )
