"""
icd10_mapper.py - maps patient words to WHO ICD-10 codes (with fuzzy matching).

Owner: Member 1 (Data & Standards).

What it does: reads the patient's words (chief complaint, symptoms, history)
and suggests matching ICD-10 codes for the doctor to confirm.

IMPORTANT
  * This is a keyword lookup, NOT a diagnosis.
  * Codes are WHO ICD-10 (the version India's ABDM uses), which has at most
    4 characters (e.g. J45.9). This is different from the US "ICD-10-CM"
    (e.g. J45.909). Before the final demo, have a clinician or mentor spot-
    check the codes in the WHO browser: https://icd.who.int/browse10/2019/en
  * Negation is NOT understood: "no chest pain" still matches chest pain.
    The doctor must read the actual history.

Coverage: 5 WHO chapters (IV Endocrine, IX Circulatory, X Respiratory,
XI Digestive, XIII Musculoskeletal) plus chapter XVIII (symptoms and signs)
for common complaints that have no diagnosis yet.

Fuzzy matching: a keyword of 5+ letters also matches a close misspelling
(e.g. "diabetis" -> diabetes), using Python's built-in difflib.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

FUZZY_THRESHOLD = 0.85   # 1.0 = identical. Lower = more forgiving, more false matches.
MIN_FUZZY_LENGTH = 5     # shorter keywords (bp, gas, cold) must match exactly.

# (code, name, WHO chapter, keywords). Keywords are lowercase.
_ENTRIES: List[Tuple[str, str, str, List[str]]] = [
    # ===== Chapter IV - Endocrine, nutritional and metabolic =====
    ("E03.9", "Hypothyroidism, unspecified", "IV",
     ["hypothyroidism", "underactive thyroid", "low thyroid"]),
    ("E05.9", "Thyrotoxicosis, unspecified", "IV",
     ["hyperthyroidism", "thyrotoxicosis", "overactive thyroid"]),
    ("E10.9", "Type 1 diabetes mellitus without complications", "IV",
     ["type 1 diabetes", "juvenile diabetes"]),
    ("E11.9", "Type 2 diabetes mellitus without complications", "IV",
     ["type 2 diabetes", "diabetes", "diabetic", "sugar", "high sugar",
      "prameha", "madhumeha", "hba1c"]),
    ("E11.8", "Type 2 diabetes mellitus with unspecified complications", "IV",
     ["uncontrolled diabetes", "diabetic complications", "diabetic foot"]),
    ("E66.9", "Obesity, unspecified", "IV",
     ["obesity", "overweight", "sthaulya", "weight gain"]),
    ("E78.5", "Hyperlipidaemia, unspecified", "IV",
     ["high cholesterol", "hyperlipidemia", "high lipids", "dyslipidemia"]),
    ("E55.9", "Vitamin D deficiency, unspecified", "IV",
     ["vitamin d deficiency", "low vitamin d"]),
    ("E53.8", "Deficiency of other specified B group vitamins", "IV",
     ["vitamin b12 deficiency", "b12 deficiency", "low b12"]),
    ("E86", "Volume depletion", "IV",
     ["dehydration", "dehydrated"]),

    # ===== Chapter IX - Circulatory =====
    ("I10", "Essential (primary) hypertension", "IX",
     ["hypertension", "high blood pressure", "high bp", "elevated bp", "bp"]),
    ("I20.9", "Angina pectoris, unspecified", "IX",
     ["angina", "hridshoola"]),
    ("I21.9", "Acute myocardial infarction, unspecified", "IX",
     ["heart attack", "myocardial infarction"]),
    ("I25.9", "Chronic ischaemic heart disease, unspecified", "IX",
     ["ischemic heart disease", "coronary artery disease", "heart disease"]),
    ("I48", "Atrial fibrillation and flutter", "IX",
     ["atrial fibrillation", "afib"]),
    ("I49.9", "Cardiac arrhythmia, unspecified", "IX",
     ["arrhythmia", "irregular heartbeat", "irregular pulse"]),
    ("I50.9", "Heart failure, unspecified", "IX",
     ["heart failure", "cardiac failure", "swollen ankles"]),
    ("I64", "Stroke, not specified as haemorrhage or infarction", "IX",
     ["stroke", "paralysis", "face drooping"]),
    ("I83.9", "Varicose veins of lower extremities without ulcer or inflammation", "IX",
     ["varicose veins", "varicose vein"]),
    ("I95.9", "Hypotension, unspecified", "IX",
     ["low blood pressure", "low bp", "hypotension"]),

    # ===== Chapter X - Respiratory =====
    ("J00", "Acute nasopharyngitis (common cold)", "X",
     ["common cold", "cold", "running nose", "runny nose", "pratishyaya"]),
    ("J01.9", "Acute sinusitis, unspecified", "X",
     ["sinusitis", "sinus pain", "sinus infection"]),
    ("J02.9", "Acute pharyngitis, unspecified", "X",
     ["pharyngitis", "sore throat", "throat pain"]),
    ("J03.9", "Acute tonsillitis, unspecified", "X",
     ["tonsillitis", "swollen tonsils"]),
    ("J06.9", "Acute upper respiratory infection, unspecified", "X",
     ["upper respiratory infection", "urti", "viral infection"]),
    ("J18.9", "Pneumonia, unspecified", "X",
     ["pneumonia", "chest infection"]),
    ("J20.9", "Acute bronchitis, unspecified", "X",
     ["bronchitis", "chest congestion"]),
    ("J30.4", "Allergic rhinitis, unspecified", "X",
     ["allergic rhinitis", "hay fever", "sneezing", "dust allergy"]),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified", "X",
     ["copd", "chronic bronchitis", "emphysema"]),
    ("J45.9", "Asthma, unspecified", "X",
     ["asthma", "wheezing", "tamaka shwasa"]),

    # ===== Chapter XI - Digestive =====
    ("K21.9", "Gastro-oesophageal reflux disease without oesophagitis", "XI",
     ["acidity", "acid reflux", "heartburn", "burning sensation in stomach",
      "burning chest", "amlapitta", "sour burp", "gerd"]),
    ("K25.9", "Gastric ulcer, unspecified", "XI",
     ["gastric ulcer", "stomach ulcer"]),
    ("K29.7", "Gastritis, unspecified", "XI",
     ["gastritis", "stomach irritation", "epigastric pain", "burning stomach"]),
    ("K30", "Dyspepsia", "XI",
     ["indigestion", "dyspepsia", "upset stomach"]),
    ("K52.9", "Noninfective gastroenteritis and colitis, unspecified", "XI",
     ["gastroenteritis", "diarrhea", "diarrhoea", "loose motions", "loose stools"]),
    ("K58.9", "Irritable bowel syndrome without diarrhoea", "XI",
     ["irritable bowel", "ibs"]),
    ("K59.0", "Constipation", "XI",
     ["constipation", "hard stool", "irregular bowel", "vibandha"]),
    ("K64.9", "Haemorrhoids, unspecified", "XI",
     ["haemorrhoids", "hemorrhoids", "piles"]),
    ("K76.0", "Fatty (change of) liver, not elsewhere classified", "XI",
     ["fatty liver"]),
    ("K80.2", "Calculus of gallbladder without cholecystitis", "XI",
     ["gallstones", "gall stones", "gallbladder stones"]),

    # ===== Chapter XIII - Musculoskeletal =====
    ("M06.9", "Rheumatoid arthritis, unspecified", "XIII",
     ["rheumatoid arthritis"]),
    ("M10.9", "Gout, unspecified", "XIII",
     ["gout", "high uric acid"]),
    ("M13.9", "Arthritis, unspecified", "XIII",
     ["arthritis", "swollen joints"]),
    ("M17.9", "Gonarthrosis (knee osteoarthritis), unspecified", "XIII",
     ["knee osteoarthritis", "osteoarthritis of knee"]),
    ("M19.9", "Arthrosis (osteoarthritis), unspecified", "XIII",
     ["osteoarthritis", "sandhigata vata", "joint stiffness", "crepitus"]),
    ("M25.5", "Pain in joint", "XIII",
     ["joint pain", "knee pain", "shoulder pain", "ankle pain", "hip pain"]),
    ("M54.2", "Cervicalgia (neck pain)", "XIII",
     ["neck pain", "cervical pain", "stiff neck pain"]),
    ("M54.3", "Sciatica", "XIII",
     ["sciatica", "gridhrasi", "pain radiating to leg"]),
    ("M54.5", "Low back pain", "XIII",
     ["low back pain", "back pain", "lumbar pain", "backache", "kati shoola"]),
    ("M75.0", "Adhesive capsulitis of shoulder (frozen shoulder)", "XIII",
     ["frozen shoulder"]),
    ("M79.1", "Myalgia", "XIII",
     ["body ache", "muscle pain", "myalgia", "angamarda"]),
    ("M81.9", "Osteoporosis, unspecified", "XIII",
     ["osteoporosis", "weak bones"]),

    # ===== Chapter XVIII - Symptoms and signs (no diagnosis yet) =====
    ("R00.2", "Palpitations", "XVIII",
     ["palpitations", "racing heart", "heart racing"]),
    ("R05", "Cough", "XVIII",
     ["cough", "kasa"]),
    ("R06.0", "Dyspnoea (breathlessness)", "XVIII",
     ["shortness of breath", "breathlessness", "difficulty breathing", "shwasa"]),
    ("R07.4", "Chest pain, unspecified", "XVIII",
     ["chest pain", "chest pressure", "chest tightness", "radiating chest pain"]),
    ("R10.4", "Other and unspecified abdominal pain", "XVIII",
     ["abdominal pain", "stomach pain", "stomach ache", "belly pain"]),
    ("R11", "Nausea and vomiting", "XVIII",
     ["nausea", "vomiting", "feeling sick"]),
    ("R14", "Flatulence and related conditions", "XVIII",
     ["bloating", "gas", "flatulence"]),
    ("R42", "Dizziness and giddiness", "XVIII",
     ["dizziness", "giddiness", "vertigo", "lightheaded"]),
    ("R50.9", "Fever, unspecified", "XVIII",
     ["fever", "pyrexia", "high temp", "jwara", "chills"]),
    ("R51", "Headache", "XVIII",
     ["headache", "migraine", "head pain", "shirashoola"]),
    ("R53", "Malaise and fatigue", "XVIII",
     ["fatigue", "tiredness", "weakness", "lethargy", "daurbalya", "klama"]),
    ("R73.9", "Hyperglycaemia, unspecified", "XVIII",
     ["high blood sugar", "hyperglycemia"]),
]

ICD10_DATABASE: List[Dict] = [
    {"code": c, "name": n, "chapter": ch, "keywords": kws}
    for c, n, ch, kws in _ENTRIES
]

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> List[Tuple[str, int, int]]:
    """Split text into (word, start, end) pieces."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _find_matches(text_lower: str, keyword: str, allow_fuzzy: bool = True) -> List[Tuple[int, int, str, float]]:
    """All places where `keyword` appears. Returns (start, end, type, score)."""
    found: List[Tuple[int, int, str, float]] = []

    exact = re.compile(r"(?<!\w)" + r"\s+".join(re.escape(w) for w in keyword.split()) + r"(?!\w)")
    for m in exact.finditer(text_lower):
        found.append((m.start(), m.end(), "exact", 1.0))
    if found or not allow_fuzzy or len(keyword) < MIN_FUZZY_LENGTH:
        return found

    # fuzzy: compare with every run of the same number of words in the text
    toks = _tokens(text_lower)
    n = len(keyword.split())
    for i in range(len(toks) - n + 1):
        window = toks[i:i + n]
        phrase = " ".join(w for w, _, _ in window)
        if abs(len(phrase) - len(keyword)) > 3:
            continue
        score = SequenceMatcher(None, keyword, phrase).ratio()
        if score >= FUZZY_THRESHOLD:
            found.append((window[0][1], window[-1][2], "fuzzy", round(score, 2)))
    return found


def map_text_to_icd10(text: str, fuzzy: bool = True) -> List[Dict[str, str]]:
    """Return the unique ICD-10 matches found in `text`.

    Each result: {"code", "name", "matched_keyword", "chapter", "match_type"}.
    match_type is "exact" or "fuzzy" (a close misspelling).
    """
    if not text:
        return []
    text_lower = text.lower()

    candidates = []   # (start, end, db_index, keyword, type, score)
    for idx, item in enumerate(ICD10_DATABASE):
        best = None
        for kw in item["keywords"]:
            for start, end, mtype, score in _find_matches(text_lower, kw, fuzzy):
                rank = (mtype == "exact", end - start)
                if best is None or rank > best[0]:
                    best = (rank, (start, end, idx, kw, mtype, score))
        if best:
            candidates.append(best[1])

    # If a match sits fully inside a longer match for another code
    # (e.g. "arthritis" inside "rheumatoid arthritis"), keep the longer one.
    candidates.sort(key=lambda c: c[1] - c[0], reverse=True)
    kept = []
    for cand in candidates:
        inside_longer = any(k[0] <= cand[0] and cand[1] <= k[1] for k in kept)
        if not inside_longer:
            kept.append(cand)

    kept.sort(key=lambda c: c[2])   # stable order: same as the database
    results = []
    for start, end, idx, kw, mtype, score in kept:
        item = ICD10_DATABASE[idx]
        results.append({
            "code": item["code"],
            "name": item["name"],
            "matched_keyword": kw,
            "chapter": item["chapter"],
            "match_type": mtype,
        })
    return results


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


def coverage_summary() -> Dict[str, int]:
    """How many codes per WHO chapter."""
    summary: Dict[str, int] = {}
    for item in ICD10_DATABASE:
        summary[item["chapter"]] = summary.get(item["chapter"], 0) + 1
    return summary


if __name__ == "__main__":
    # Quick self-test: python -m app.icd10_mapper
    summary = coverage_summary()
    core = sum(v for k, v in summary.items() if k != "XVIII")
    print("Codes per chapter:", summary)
    print(f"Total codes: {len(ICD10_DATABASE)} (core 5 chapters: {core})")

    tests = [
        "I have chest pain, high blood pressure and gastritis",
        "diabetis and hypertention since 5 years",
        "rheumatoid arthritis with knee pain",
        "severe headdache and feaver",
    ]
    for text in tests:
        print("\nTEXT:", text)
        for m in map_text_to_icd10(text):
            print(f"  {m['code']:7} {m['name']} | matched '{m['matched_keyword']}' ({m['match_type']})")