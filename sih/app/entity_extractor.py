"""
Pulls structured medical entities out of raw OCR text:
- medicines + dosage/frequency
- diagnoses / conditions
- lab test values
- dates

Starter version uses regex + light heuristics, which is a reasonable
baseline for a hackathon MVP. Swap in a trained spaCy NER model or a
medical NER model (e.g. scispaCy / a biomedical BERT) once you have
labeled data or more time — the extract_entities() function is the
seam to plug that in later without touching the rest of the pipeline.
"""
import re
from dateutil import parser as dateparser

# Common dosage patterns: "500mg", "5 ml", "1-0-1", "twice daily"
DOSAGE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g|IU|units?)\b", re.IGNORECASE
)
FREQUENCY_PATTERN = re.compile(
    r"\b(once|twice|thrice|\d+-\d+-\d+|OD|BD|TDS|QID|SOS|HS)\b", re.IGNORECASE
)
DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|"
    r"Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
    re.IGNORECASE,
)

# Small starter vocab — extend from your team's ontology / Ayush drug lists
COMMON_DIAGNOSIS_KEYWORDS = [
    "diabetes", "hypertension", "asthma", "fever", "infection", "anemia",
    "arthritis", "migraine", "tuberculosis", "hypothyroidism", "vata",
    "pitta", "kapha", "jwara", "prameha",
]

LAB_TEST_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z\s]{2,30}?)[:\s]+(\d+(\.\d+)?)\s?(mg/dl|g/dl|%|mmHg|"
    r"mmol/l|/mm3|IU/L)?\b",
    re.IGNORECASE,
)


def extract_medicines(text: str) -> list[dict]:
    lines = text.split("\n")
    medicines = []
    for line in lines:
        dosage_match = DOSAGE_PATTERN.search(line)
        if dosage_match:
            freq_match = FREQUENCY_PATTERN.search(line)
            medicines.append({
                "raw_line": line.strip(),
                "dosage": dosage_match.group(0),
                "frequency": freq_match.group(0) if freq_match else None,
            })
    return medicines


def extract_dates(text: str) -> list[str]:
    found = DATE_PATTERN.findall(text)
    # findall with groups returns tuples for some matches; re-search cleanly instead
    dates = [m.group(0) for m in DATE_PATTERN.finditer(text)]
    normalized = []
    for d in dates:
        try:
            normalized.append(dateparser.parse(d, fuzzy=True).date().isoformat())
        except (ValueError, OverflowError):
            continue
    return normalized


def extract_diagnoses(text: str) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in COMMON_DIAGNOSIS_KEYWORDS if kw in text_lower]


def extract_lab_values(text: str) -> list[dict]:
    results = []
    for match in LAB_TEST_PATTERN.finditer(text):
        name, value, _, unit = match.groups()
        name = name.strip()
        if len(name.split()) <= 4:  # filter out noisy long false-positive matches
            results.append({"test": name, "value": value, "unit": unit})
    return results


def extract_entities(ocr_text: str) -> dict:
    """Main entry point: raw OCR text -> structured clinical entities."""
    return {
        "medicines": extract_medicines(ocr_text),
        "diagnoses": extract_diagnoses(ocr_text),
        "dates": extract_dates(ocr_text),
        "lab_values": extract_lab_values(ocr_text),
    }
