"""
Shared clinical-history schema.

This is the contract every other MediKiosk module should build against:
- the OCR/document module should be able to merge its extracted fields into
  the same shape,
- the FHIR/ABDM bridge maps this into a Bundle,
- the frontend renders it on the physician dashboard.

Keep this file the single source of truth for field names. If you need a new
field, add it here first and tell the rest of the team before you start
emitting it from another module.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

TriageLevel = Literal["routine", "soon", "urgent", "emergency"]


class HPI(BaseModel):
    """History of Present Illness, captured via the SOCRATES framework."""

    site: Optional[str] = None
    onset: Optional[str] = None
    character: Optional[str] = None
    radiation: Optional[str] = None
    associated_symptoms: List[str] = Field(default_factory=list)
    timing: Optional[str] = None
    exacerbating_relieving: Optional[str] = None
    severity: Optional[str] = None


class AyushAssessment(BaseModel):
    """
    Lightweight Dashavidha-Pariksha-inspired intake fields.

    PLACEHOLDER SCOPE NOTE: these fields and the questions that fill them
    (see app/prompts.py) have NOT been validated by a practicing Ayurvedic
    physician. Treat this as a structural draft, not a clinically correct
    assessment, until someone with that training reviews it. Say this
    explicitly if you demo it to judges.
    """

    prakriti: Optional[str] = None
    vikriti: Optional[str] = None
    agni: Optional[str] = None
    koshtha: Optional[str] = None
    ahara_vihara: Optional[str] = None
    nidana: Optional[str] = None
    samprapti: Optional[str] = None


class ClinicalHistory(BaseModel):
    chief_complaint: Optional[str] = None
    hpi: HPI = Field(default_factory=HPI)
    past_medical_surgical_history: List[str] = Field(default_factory=list)
    # current_medications / drug_allergy_history used to be one merged
    # "drug_allergy_history" free-text field. That was a real ambiguity risk
    # — a physician reading "metformin" in it couldn't tell whether the
    # patient takes it or reacts to it — so as of 2026-09-07 they're split.
    # If you're looking at old session data or a downstream consumer still
    # expecting the merged shape, that's the bug this split fixes, not a
    # regression. Field descriptions below are sent to the LLM as part of
    # its schema, so keep them purely instructional — no changelog text.
    current_medications: List[str] = Field(
        default_factory=list,
        description="Medicines the patient is CURRENTLY TAKING. Never put an allergy here — "
        "see drug_allergy_history for that.",
    )
    drug_allergy_history: List[str] = Field(
        default_factory=list,
        description="Medicines the patient is ALLERGIC TO (or has had a bad reaction to). "
        "Never put a currently-taken medication here — see current_medications for that.",
    )
    family_history: List[str] = Field(default_factory=list)
    personal_history: Optional[str] = None
    review_of_systems: List[str] = Field(default_factory=list)
    ayush: Optional[AyushAssessment] = None


class TurnResult(BaseModel):
    """
    What one dialogue turn produces. The model is asked to re-derive the
    FULL current history from the whole transcript each turn (not emit a
    patch) — that avoids merge-conflict bugs and is cheap enough at
    single-conversation scale.
    """

    history: ClinicalHistory
    acknowledgement: Optional[str] = Field(
        default=None,
        description="One short, empathetic sentence responding to what the patient just said.",
    )
    next_question: str = Field(
        default="Can you tell me more about that?",
        min_length=1,
        description="The single next question to ask, as a natural-language string. "
        "This must NEVER be empty, under any circumstance — not even when complete "
        "or red_flag is true. If the interview is ending, put a short one-sentence "
        "closing remark here instead of a question (e.g. 'Thank you, that's "
        "everything we need for now.'). There is no situation where leaving this "
        "blank is the correct response.",
    )
    complete: bool = Field(
        default=False,
        description="True once chief complaint, HPI, past history, and drug/allergy history are reasonably captured.",
    )
    red_flag: bool = Field(
        default=False,
        description="True if the patient described an emergency symptom that should bypass the normal queue.",
    )
    red_flag_reason: Optional[str] = None
    triage_level: Optional[TriageLevel] = Field(
        default=None,
        description="Your best-effort read of how urgently this patient should be seen, given "
        "everything gathered so far: "
        "'emergency' = should always coincide with red_flag=true, stop the normal interview; "
        "'urgent' = not an emergency, but should be prioritized over a routine queue "
        "(e.g. high fever with vomiting, significant pain, a worsening chronic condition); "
        "'soon' = should be seen this visit without excessive delay, but not prioritized above "
        "others; "
        "'routine' = a routine complaint with no urgency signals. "
        "Leave this null until you have at least a chief complaint and a rough sense of "
        "severity — don't guess from nothing on turn one. This is advisory information for "
        "the queue, separate from red_flag: red_flag alone still controls whether the "
        "interview stops.",
    )

    @classmethod
    def from_raw_lenient(cls, raw: str) -> TurnResult:
        """
        Parses JSON and leniently handles cases where the backend outputs an empty
        `next_question`, which would otherwise fail the `min_length=1` validation.
        It also handles cases where smaller local models flatten the JSON output 
        instead of properly nesting clinical fields inside the 'history' object,
        or when they output empty strings instead of empty lists.
        """
        import json
        data = json.loads(raw)
        
        # 1. Handle missing/empty next_question
        nq = data.get("next_question")
        if nq == "" or nq is None:
            data["next_question"] = "DUMMY_EMPTY"
            
        # 2. Re-nest flattened history fields if the model forgot the 'history' wrapper
        if "history" not in data:
            history_fields = [
                "chief_complaint", "hpi", "past_medical_surgical_history",
                "current_medications", "drug_allergy_history", "family_history",
                "personal_history", "review_of_systems", "ayush"
            ]
            history_obj = {}
            for field in history_fields:
                if field in data:
                    history_obj[field] = data.pop(field)
            data["history"] = history_obj
            
        # 2b. Re-nest flattened HPI fields if the model put them at the root
        hpi_fields = [
            "site", "onset", "character", "radiation", "associated_symptoms",
            "timing", "exacerbating_relieving", "severity"
        ]
        hpi_obj = data["history"].get("hpi", {}) if isinstance(data.get("history"), dict) else {}
        found_flattened_hpi = False
        for field in hpi_fields:
            if field in data:
                hpi_obj[field] = data.pop(field)
                found_flattened_hpi = True
        
        if found_flattened_hpi and isinstance(data.get("history"), dict):
            data["history"]["hpi"] = hpi_obj

        # 2c. Re-nest flattened Ayush fields if the model put them at the root
        ayush_fields = [
            "prakriti", "vikriti", "agni", "koshtha", "ahara_vihara", "nidana", "samprapti"
        ]
        ayush_obj = data["history"].get("ayush", {}) if isinstance(data.get("history"), dict) and data["history"].get("ayush") is not None else {}
        found_flattened_ayush = False
        for field in ayush_fields:
            if field in data:
                ayush_obj[field] = data.pop(field)
                found_flattened_ayush = True
        
        if found_flattened_ayush and isinstance(data.get("history"), dict):
            data["history"]["ayush"] = ayush_obj

        # 3. Fix list fields that were incorrectly returned as empty strings
        if "history" in data and isinstance(data["history"], dict):
            h = data["history"]
            list_fields = [
                "past_medical_surgical_history", "current_medications",
                "drug_allergy_history", "family_history", "review_of_systems"
            ]
            for lf in list_fields:
                if h.get(lf) == "" or h.get(lf) == "none" or h.get(lf) == "None":
                    h[lf] = []
                elif isinstance(h.get(lf), str):
                    h[lf] = [h[lf]]
            
            if "hpi" in h and isinstance(h["hpi"], dict):
                if h["hpi"].get("associated_symptoms") == "" or h["hpi"].get("associated_symptoms") in ("none", "None"):
                    h["hpi"]["associated_symptoms"] = []
                elif isinstance(h["hpi"].get("associated_symptoms"), str):
                    h["hpi"]["associated_symptoms"] = [h["hpi"]["associated_symptoms"]]

        obj = cls.model_validate(data)
        
        if obj.next_question == "DUMMY_EMPTY":
            obj.next_question = ""
            
        return obj
