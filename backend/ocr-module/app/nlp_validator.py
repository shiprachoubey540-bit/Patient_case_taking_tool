"""Task 2 - spaCy clinical NLP validator.

Finds medicines, doses, frequencies and durations in OCR text, then checks
the medicines against a SMALL DEMO interaction table. This table is for the
project demo only. It is not a complete or clinically approved database.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import spacy

# --- 1. Reference lists (you can edit these) --------------------------------
MEDICINES: List[str] = [
    "paracetamol", "ibuprofen", "aspirin", "diclofenac", "metformin",
    "glimepiride", "insulin", "amlodipine", "atenolol", "losartan",
    "enalapril", "telmisartan", "atorvastatin", "rosuvastatin", "warfarin",
    "clopidogrel", "omeprazole", "pantoprazole", "ranitidine", "domperidone",
    "amoxicillin", "azithromycin", "ciprofloxacin", "cetirizine",
    "salbutamol", "prednisolone", "furosemide", "spironolactone",
    "ondansetron", "levothyroxine",
]

# (drug A, drug B, severity, note)
INTERACTIONS = [
    ("warfarin", "aspirin", "high", "Higher bleeding risk"),
    ("warfarin", "ibuprofen", "high", "Higher bleeding risk"),
    ("warfarin", "diclofenac", "high", "Higher bleeding risk"),
    ("clopidogrel", "omeprazole", "moderate", "Omeprazole can weaken clopidogrel"),
    ("aspirin", "ibuprofen", "moderate", "Ibuprofen can block aspirin's heart protection"),
    ("enalapril", "spironolactone", "high", "Risk of high potassium"),
    ("losartan", "spironolactone", "high", "Risk of high potassium"),
    ("atenolol", "insulin", "moderate", "Can hide signs of low blood sugar"),
    ("metformin", "ciprofloxacin", "moderate", "Blood sugar may swing up or down"),
    ("atorvastatin", "azithromycin", "low", "Small increase in muscle-injury risk"),
]

# --- 2. Build the spaCy pipeline (blank English + rules, no download) -------
_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        patterns = [{"label": "MEDICINE", "pattern": [{"LOWER": name}]} for name in MEDICINES]
        patterns += [
            {"label": "DOSAGE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["mg", "mcg", "g", "ml", "iu"]}}]},
            {"label": "DOSAGE", "pattern": [{"TEXT": {"REGEX": r"^\d+(\.\d+)?(mg|mcg|g|ml|iu)$"}}]},
            {"label": "FREQUENCY", "pattern": [{"LOWER": {"IN": ["od", "bd", "bid", "tds", "tid", "qid", "hs", "sos", "prn"]}}]},
            {"label": "DURATION", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months"]}}]},
        ]
        ruler.add_patterns(patterns)
        _nlp = nlp
    return _nlp


# --- 3. Main functions ------------------------------------------------------
def extract_entities(text: str) -> Dict[str, List[str]]:
    doc = get_nlp()(text)
    found: Dict[str, List[str]] = {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
    for ent in doc.ents:
        value = ent.text.lower() if ent.label_ == "MEDICINE" else ent.text
        if value not in found[ent.label_]:
            found[ent.label_].append(value)
    return found


def check_interactions(medicines: List[str]) -> List[dict]:
    names = {m.lower() for m in medicines}
    alerts = []
    for a, b, severity, note in INTERACTIONS:
        if a in names and b in names:
            alerts.append({"drug_a": a, "drug_b": b, "severity": severity, "note": note})
    return alerts


def validate_prescription_text(text: str) -> dict:
    """Text in, structured result out. This is the function other modules call."""
    entities = extract_entities(text)
    meds = entities["MEDICINE"]
    alerts = check_interactions(meds)
    return {
        "medicines": meds,
        "dosages": entities["DOSAGE"],
        "frequencies": entities["FREQUENCY"],
        "durations": entities["DURATION"],
        "interaction_alerts": alerts,
        "has_interaction": len(alerts) > 0,
        "pairs_checked": len(list(combinations(meds, 2))),
        "disclaimer": "Demo reference table only - doctor must confirm.",
    }


if __name__ == "__main__":
    import json

    sample = "Tab Warfarin 5mg OD x 5 days, Tab Aspirin 75mg OD, Tab Paracetamol 650mg BD"
    print(json.dumps(validate_prescription_text(sample), indent=2))