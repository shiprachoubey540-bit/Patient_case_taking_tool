"""
Shared schema for extracted documents.

Mirrors the design of the Converse module's schema.py: this is the contract
the backend/integration lead's FHIR bridge and the physician dashboard
should build against. Keep field names stable; add fields here first if you
need new ones.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

DOCUMENT_TYPES = ("prescription", "lab_report", "discharge_summary", "other")


class Medication(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    confidence: float = Field(
        default=1.0,
        description="0-1. Lower for anything uncertain, e.g. read from handwriting.",
    )


class Investigation(BaseModel):
    name: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    abnormal: Optional[bool] = Field(
        default=None,
        description="True/false only when a reference range is visible on the document "
        "and the value falls outside it. Null if it can't be determined — never guess.",
    )
    confidence: float = 1.0


class Vitals(BaseModel):
    """
    Bedside vitals (often under an "O/E" — on examination — heading) don't
    fit the lab-style name/value/unit/reference_range shape Investigation
    expects, so they need their own field. Added after live testing showed
    the model correctly declining to force-fit "BP 110/70" into
    Investigation, but then having nowhere else to put it — it was silently
    dropped instead of surfaced.
    """

    blood_pressure: Optional[str] = None
    pulse_rate: Optional[str] = None
    temperature: Optional[str] = None
    respiratory_rate: Optional[str] = None
    spo2: Optional[str] = None


class ExtractedDocument(BaseModel):
    document_type: Optional[str] = Field(
        default=None, description="One of: " + ", ".join(DOCUMENT_TYPES)
    )
    document_date: Optional[str] = Field(
        default=None, description="Date as written on the document, verbatim — do not normalize or guess a format."
    )
    presenting_complaints: List[str] = Field(
        default_factory=list,
        description="Symptoms the patient reported, usually under a 'c/o' (complains of) heading — "
        "e.g. 'giddiness', 'restlessness'. These are NOT diagnoses. Added after live testing showed "
        "the model correctly reading c/o content but having nowhere to put it except diagnoses, which "
        "conflated 'what the patient reported' with 'what the clinician concluded'.",
    )
    diagnoses: List[str] = Field(
        default_factory=list,
        description="The clinician's actual diagnosis/impression only — usually under 'Imp:', 'Dx:', "
        "or 'Diagnosis:'. Do not put presenting symptoms here even if no separate diagnosis is written; "
        "if the document only lists symptoms with no stated impression, leave this empty and put them "
        "in presenting_complaints instead.",
    )
    vitals: Optional[Vitals] = None
    medications: List[Medication] = Field(default_factory=list)
    investigations: List[Investigation] = Field(default_factory=list)
    procedures_surgeries: List[str] = Field(default_factory=list)
    raw_text_excerpt: Optional[str] = Field(
        default=None,
        description="A short excerpt (a few lines) of what was actually read, so a physician can "
        "sanity-check the extraction against the source at a glance.",
    )
    overall_confidence: float = Field(
        default=1.0, description="0-1, this document's extraction as a whole."
    )
    needs_review: bool = Field(
        default=False,
        description="True if handwriting, poor image quality, or any low-confidence field was involved.",
    )
    review_reason: Optional[str] = None


class StoredDocument(BaseModel):
    id: str
    session_id: Optional[str] = None
    filename: Optional[str] = None
    uploaded_at: float
    backend: str
    extracted: ExtractedDocument
