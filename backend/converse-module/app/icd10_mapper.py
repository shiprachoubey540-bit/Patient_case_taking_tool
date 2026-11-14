"""
icd10_mapper.py - Standard WHO ICD-10 code auto-mapping for OPD clinical intake.

Owner: Member 1 (Data & Standards).

What it does: reads the patient's words (chief complaint, symptoms, past
history) and returns the matching ICD-10 codes.

IMPORTANT: this is a keyword lookup, NOT a diagnosis. The codes are
suggestions for the doctor to confirm.

Phase 1 note: this file was repaired. It previously contained PowerShell
text (the lines starting with @' and ending with '@ | Set-Content) which
made Python fail with a SyntaxError. The code list is unchanged; expansion
to 45+ codes and fuzzy matching come in a later phase.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Each entry: ICD-10 code, readable name, and keywords that trigger it.
ICD10_DATABASE = [
    # --- Gastrointestinal / Amlapitta ---
    {
        "code": "K21.9",
        "name": "Gastro-esophageal reflux disease without esophagitis (GERD)",
        "keywords": ["acidity", "acid reflux", "heartburn", "burning sensation in stomach",
                     "burning chest", "amlapitta", "sour burp", "gerd", "indigestion", "dyspepsia"],
    },
    {
        "code": "K59.00",
        "name": "Constipation, unspecified (Vibandha)",
        "keywords": ["constipation", "hard stool", "irregular bowel", "vibandha", "bloating", "gas"],
    },
    {
        "code": "K29.70",
        "name": "Gastritis, unspecified",
        "keywords": ["gastritis", "stomach irritation", "epigastric pain", "burning stomach"],
    },
    # --- Endocrine & Metabolic / Prameha ---
    {
        "code": "E11.9",
        "name": "Type 2 diabetes mellitus without complications (Prameha)",
        "keywords": ["type 2 diabetes", "diabetes", "diabetic", "sugar", "high sugar",
                     "prameha", "madhumeha", "hba1c"],
    },
    {
        "code": "E11.65",
        "name": "Type 2 diabetes mellitus with hyperglycemia",
        "keywords": ["uncontrolled diabetes", "severe hyperglycemia", "high blood sugar"],
    },
    {
        "code": "E66.9",
        "name": "Obesity, unspecified (Sthaulya)",
        "keywords": ["obesity", "overweight", "sthaulya", "weight gain"],
    },
    # --- Cardiovascular ---
    {
        "code": "I10",
        "name": "Essential (primary) hypertension",
        "keywords": ["hypertension", "high blood pressure", "high bp", "elevated bp", "bp"],
    },
    {
        "code": "I20.9",
        "name": "Angina pectoris, unspecified (Chest discomfort / Hridshoola)",
        "keywords": ["angina", "chest pain", "chest pressure", "radiating chest pain", "hridshoola"],
    },
    # --- Musculoskeletal ---
    {
        "code": "M19.90",
        "name": "Unspecified osteoarthritis, unspecified site (Sandhigata Vata)",
        "keywords": ["osteoarthritis", "joint pain", "knee pain", "arthritis",
                     "sandhigata vata", "joint stiffness", "crepitus"],
    },
    {
        "code": "M54.50",
        "name": "Low back pain, unspecified (Kati Shoola)",
        "keywords": ["back pain", "low back pain", "lumbar pain", "kati shoola", "sciatica", "gridhrasi"],
    },
    {
        "code": "M79.1",
        "name": "Myalgia (Body ache / Angamarda)",
        "keywords": ["body ache", "muscle pain", "myalgia", "angamarda"],
    },
    # --- Respiratory ---
    {
        "code": "J00",
        "name": "Acute nasopharyngitis (Common cold / Pratishyaya)",
        "keywords": ["cold", "cough", "running nose", "pratishyaya", "kasa", "sore throat"],
    },
    {
        "code": "J45.909",
        "name": "Unspecified asthma, uncomplicated (Tamaka Shwasa)",
        "keywords": ["asthma", "wheezing", "breathlessness", "shwasa", "tamaka shwasa", "shortness of breath"],
    },
    # --- General symptoms ---
    {
        "code": "R50.9",
        "name": "Fever, unspecified (Jwara)",
        "keywords": ["fever", "pyrexia", "high temp", "jwara", "chills"],
    },
    {
        "code": "R53.83",
        "name": "Other fatigue (Klama / Daurbalya)",
        "keywords": ["fatigue", "tiredness", "weakness", "lethargy", "daurbalya", "klama"],
    },
    {
        "code": "R51.9",
        "name": "Headache, unspecified (Shirashoola)",
        "keywords": ["headache", "migraine", "head pain", "shirashoola"],
    },
]


def map_text_to_icd10(text: str) -> List[Dict[str, str]]:
    """Return the unique ICD-10 matches found in `text`.

    Keywords must match as whole words/phrases, so "gas" does not match
    inside "gastritis" and "bp" does not match inside another word.
    """
    if not text:
        return []

    text_lower = text.lower()
    matches: List[Dict[str, str]] = []
    seen_codes = set()

    for item in ICD10_DATABASE:
        for kw in item["keywords"]:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                if item["code"] not in seen_codes:
                    matches.append({
                        "code": item["code"],
                        "name": item["name"],
                        "matched_keyword": kw,
                    })
                    seen_codes.add(item["code"])
                break  # one match is enough for this code

    return matches


def extract_icd10_from_history(history_dict: dict) -> List[Dict[str, str]]:
    """Collect text from the clinical-history fields and map it to ICD-10 codes."""
    all_texts: List[str] = []

    if history_dict.get("chief_complaint"):
        all_texts.append(str(history_dict["chief_complaint"]))

    past = history_dict.get("past_medical_surgical_history", [])
    if isinstance(past, list):
        all_texts.extend(str(p) for p in past)

    hpi = history_dict.get("hpi", {})
    if isinstance(hpi, dict):
        if hpi.get("character"):
            all_texts.append(str(hpi["character"]))
        assoc = hpi.get("associated_symptoms", [])
        if isinstance(assoc, list):
            all_texts.extend(str(a) for a in assoc)

    ayush = history_dict.get("ayush")
    if isinstance(ayush, dict) and ayush.get("nidana"):
        all_texts.append(str(ayush["nidana"]))

    return map_text_to_icd10(" ".join(all_texts))


if __name__ == "__main__":
    # Quick self-test: python -m app.icd10_mapper
    sample = "I have chest pain, high blood pressure and gastritis"
    for m in map_text_to_icd10(sample):
        print(m["code"], "-", m["name"], "| matched:", m["matched_keyword"])