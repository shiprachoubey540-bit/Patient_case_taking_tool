@'
"""
icd10_mapper.py - Standard WHO ICD-10 Code Auto-Mapping for OPD Clinical Intake.
Supports common presenting complaints, chronic conditions, and AYUSH correlates.
"""
from __future__ import annotations
import re
from typing import List, Dict

# Standard clinical dictionary mapping keywords/regex to ICD-10 codes
ICD10_DATABASE = [
    # Gastrointestinal / Amlapitta
    {
        "code": "K21.9",
        "name": "Gastro-esophageal reflux disease without esophagitis (GERD)",
        "keywords": ["acidity", "acid reflux", "heartburn", "burning sensation in stomach", "burning chest", "amlapitta", "sour burp", "gerd", "indigestion", "dyspepsia"]
    },
    {
        "code": "K59.00",
        "name": "Constipation, unspecified (Vibandha)",
        "keywords": ["constipation", "hard stool", "irregular bowel", "vibandha", "bloating", "gas"]
    },
    {
        "code": "K29.70",
        "name": "Gastritis, unspecified",
        "keywords": ["gastritis", "stomach irritation", "epigastric pain", "burning stomach"]
    },

    # Endocrine & Metabolic / Prameha
    {
        "code": "E11.9",
        "name": "Type 2 diabetes mellitus without complications (Prameha)",
        "keywords": ["type 2 diabetes", "diabetes", "diabetic", "sugar", "high sugar", "prameha", "madhumeha", "hba1c"]
    },
    {
        "code": "E11.65",
        "name": "Type 2 diabetes mellitus with hyperglycemia",
        "keywords": ["uncontrolled diabetes", "severe hyperglycemia", "high blood sugar"]
    },
    {
        "code": "E66.9",
        "name": "Obesity, unspecified (Sthaulya)",
        "keywords": ["obesity", "overweight", "sthaulya", "weight gain"]
    },

    # Cardiovascular / Raktachapa
    {
        "code": "I10",
        "name": "Essential (primary) hypertension",
        "keywords": ["hypertension", "high blood pressure", "high bp", "elevated bp", "bp"]
    },
    {
        "code": "I20.9",
        "name": "Angina pectoris, unspecified (Chest Discomfort / Hridshoola)",
        "keywords": ["angina", "chest pain", "chest pressure", "radiating chest pain", "hridshoola"]
    },

    # Musculoskeletal / Sandhigata Vata
    {
        "code": "M19.90",
        "name": "Unspecified osteoarthritis, unspecified site (Sandhigata Vata)",
        "keywords": ["osteoarthritis", "joint pain", "knee pain", "arthritis", "sandhigata vata", "joint stiffness", "crepitus"]
    },
    {
        "code": "M54.50",
        "name": "Low back pain, unspecified (Kati Shoola)",
        "keywords": ["back pain", "low back pain", "lumbar pain", "kati shoola", "sciatica", "gridhrasi"]
    },
    {
        "code": "M79.1",
        "name": "Myalgia (Body ache / Angamarda)",
        "keywords": ["body ache", "muscle pain", "myalgia", "angamarda"]
    },

    # Respiratory / Kasa / Shwasa
    {
        "code": "J00",
        "name": "Acute nasopharyngitis (Common cold / Pratishyaya)",
        "keywords": ["cold", "cough", "running nose", "pratishyaya", "kasa", "sore throat"]
    },
    {
        "code": "J45.909",
        "name": "Unspecified asthma, uncomplicated (Tamaka Shwasa)",
        "keywords": ["asthma", "wheezing", "breathlessness", "shwasa", "tamaka shwasa", "shortness of breath"]
    },

    # General / Fevers / Jwara
    {
        "code": "R50.9",
        "name": "Fever, unspecified (Jwara)",
        "keywords": ["fever", "pyrexia", "high temp", "jwara", "chills"]
    },
    {
        "code": "R53.83",
        "name": "Other fatigue (Klama / Daurbalya)",
        "keywords": ["fatigue", "tiredness", "weakness", "lethargy", "daurbalya", "klama"]
    },
    {
        "code": "R51.9",
        "name": "Headache, unspecified (Shirashoola)",
        "keywords": ["headache", "migraine", "head pain", "shirashoola"]
    }
]


def map_text_to_icd10(text: str) -> List[Dict[str, str]]:
    """Scan input text and return matching unique ICD-10 code objects."""
    if not text:
        return []

    text_lower = text.lower()
    matches = []
    seen_codes = set()

    for item in ICD10_DATABASE:
        for kw in item["keywords"]:
            # Match keyword as word or substring
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower) or kw in text_lower:
                if item["code"] not in seen_codes:
                    matches.append({
                        "code": item["code"],
                        "name": item["name"],
                        "matched_keyword": kw
                    })
                    seen_codes.add(item["code"])
                break

    return matches


def extract_icd10_from_history(history_dict: dict) -> List[Dict[str, str]]:
    """Extract ICD-10 codes from across all clinical history fields."""
    all_texts = []

    # Chief complaint
    if history_dict.get("chief_complaint"):
        all_texts.append(str(history_dict["chief_complaint"]))

    # Past medical history
    past = history_dict.get("past_medical_surgical_history", [])
    if isinstance(past, list):
        all_texts.extend([str(p) for p in past])

    # HPI symptoms
    hpi = history_dict.get("hpi", {})
    if isinstance(hpi, dict):
        if hpi.get("character"):
            all_texts.append(str(hpi["character"]))
        assoc = hpi.get("associated_symptoms", [])
        if isinstance(assoc, list):
            all_texts.extend([str(a) for a in assoc])

    # AYUSH nidana
    ayush = history_dict.get("ayush")
    if isinstance(ayush, dict) and ayush.get("nidana"):
        all_texts.append(str(ayush["nidana"]))

    combined = " ".join(all_texts)
    return map_text_to_icd10(combined)
'@ | Set-Content -Path "backend\converse-module\app\icd10_mapper.py" -Encoding utf8