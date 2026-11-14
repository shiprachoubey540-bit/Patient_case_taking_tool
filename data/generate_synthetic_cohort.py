"""
generate_synthetic_cohort.py - builds data/synthetic_opd_cohort_500.csv

Owner: Member 1 (Data & Standards).

Creates 500 SYNTHETIC outpatient visits that follow docs/DATA_CONTRACT.md.
Every row is invented. This is NOT real clinical data and must never be
used as clinical evidence.

Deliberate "dirty data" for the EDA exercise:
  * about 3% of vital-sign cells are left empty (missing values)
  * a handful of rows contain data-entry mistakes (outliers)

Run from the project root:
    python data/generate_synthetic_cohort.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

SEED = 42
N_ROWS = 500
MISSING_RATE = 0.03
OUTPUT_PATH = Path(__file__).resolve().parent / "synthetic_opd_cohort_500.csv"

random.seed(SEED)

# ---------------------------------------------------------------------------
# Condition catalogue. Codes match backend/converse-module/app/icd10_mapper.py.
# weight = how common it is; sev = chances of (Mild, Moderate, Severe);
# age = (mean, spread) used for the patient's age.
# ---------------------------------------------------------------------------
CONDITIONS = [
    {"code": "I10", "category": "Circulatory", "weight": 10, "age": (58, 12),
     "sev": (0.50, 0.38, 0.12),
     "symptoms": ["headache", "dizziness", "blurred vision", "palpitations", "fatigue"],
     "history": ["hypertension", "diabetes", "obesity", "smoking", "none"]},
    {"code": "I20.9", "category": "Circulatory", "weight": 5, "age": (62, 10),
     "sev": (0.20, 0.45, 0.35),
     "symptoms": ["chest pain", "chest pressure", "sweating", "shortness of breath", "nausea"],
     "history": ["hypertension", "diabetes", "high cholesterol", "smoking", "none"]},
    {"code": "J00", "category": "Respiratory", "weight": 14, "age": (30, 18),
     "sev": (0.75, 0.22, 0.03),
     "symptoms": ["cold", "cough", "running nose", "sore throat", "mild fever"],
     "history": ["allergic rhinitis", "asthma", "none", "none"]},
    {"code": "J45.9", "category": "Respiratory", "weight": 7, "age": (35, 20),
     "sev": (0.35, 0.40, 0.25),
     "symptoms": ["wheezing", "breathlessness", "cough", "chest tightness", "shortness of breath"],
     "history": ["asthma", "allergic rhinitis", "smoking", "none"]},
    {"code": "K21.9", "category": "Digestive", "weight": 9, "age": (42, 14),
     "sev": (0.65, 0.30, 0.05),
     "symptoms": ["heartburn", "acidity", "sour burp", "burning chest", "indigestion"],
     "history": ["gerd", "obesity", "none", "none"]},
    {"code": "K59.0", "category": "Digestive", "weight": 6, "age": (45, 18),
     "sev": (0.75, 0.22, 0.03),
     "symptoms": ["constipation", "hard stool", "bloating", "abdominal discomfort"],
     "history": ["irregular bowel habits", "diabetes", "none", "none"]},
    {"code": "K29.7", "category": "Digestive", "weight": 6, "age": (40, 15),
     "sev": (0.50, 0.38, 0.12),
     "symptoms": ["epigastric pain", "burning stomach", "nausea", "vomiting", "loss of appetite"],
     "history": ["gastritis", "nsaid use", "none", "none"]},
    {"code": "E11.9", "category": "Endocrine_Metabolic", "weight": 11, "age": (55, 12),
     "sev": (0.45, 0.43, 0.12),
     "symptoms": ["increased thirst", "frequent urination", "fatigue", "blurred vision", "weakness"],
     "history": ["type 2 diabetes", "hypertension", "obesity", "family history of diabetes"]},
    {"code": "E11.8", "category": "Endocrine_Metabolic", "weight": 4, "age": (58, 11),
     "sev": (0.15, 0.45, 0.40),
     "symptoms": ["increased thirst", "frequent urination", "weakness", "nausea", "blurred vision"],
     "history": ["type 2 diabetes", "hypertension", "kidney disease"]},
    {"code": "E66.9", "category": "Endocrine_Metabolic", "weight": 3, "age": (40, 13),
     "sev": (0.70, 0.27, 0.03),
     "symptoms": ["weight gain", "fatigue", "joint pain", "breathlessness on exertion"],
     "history": ["obesity", "hypertension", "prediabetes", "none"]},
    {"code": "M19.9", "category": "Musculoskeletal", "weight": 9, "age": (62, 10),
     "sev": (0.50, 0.40, 0.10),
     "symptoms": ["knee pain", "joint pain", "joint stiffness", "swelling in knee"],
     "history": ["osteoarthritis", "obesity", "previous knee injury", "none"]},
    {"code": "M54.5", "category": "Musculoskeletal", "weight": 9, "age": (45, 15),
     "sev": (0.55, 0.37, 0.08),
     "symptoms": ["low back pain", "back pain", "stiffness", "pain radiating to leg"],
     "history": ["sedentary job", "previous back injury", "none", "none"]},
    {"code": "M79.1", "category": "Musculoskeletal", "weight": 7, "age": (40, 15),
     "sev": (0.70, 0.27, 0.03),
     "symptoms": ["body ache", "muscle pain", "tiredness", "mild fever"],
     "history": ["recent viral illness", "none", "none"]},
]

SEVERITIES = ["Mild", "Moderate", "Severe"]
SEV_NUM = {"Mild": 0, "Moderate": 1, "Severe": 2}
RESPIRATORY_CODES = {"J00", "J45.9"}
CARDIAC_CODES = {"I10", "I20.9"}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_vitals(code: str, severity: str) -> dict:
    """Vital signs that get worse as severity goes up."""
    s = SEV_NUM[severity]

    spo2 = random.gauss(98 - 2.0 * s, 1.0)
    rr = random.gauss(16 + 3.0 * s, 2.0)
    pulse = random.gauss(76 + 9.0 * s, 8.0)
    sbp = random.gauss(122 + 2.0 * s, 12.0)
    temp = random.gauss(36.8 + 0.25 * s, 0.30)

    if code in RESPIRATORY_CODES:
        spo2 -= 2.0 * s
        rr += 2.0 * s
    if code in CARDIAC_CODES:
        sbp += 14 + 8 * s
        pulse += 4 * s
    if code in {"J00", "M79.1"} and s == 0 and random.random() < 0.30:
        temp += random.uniform(0.8, 1.6)          # mild fever
    if code in {"E11.8", "I20.9"} and s == 2 and random.random() < 0.25:
        sbp -= random.uniform(20, 35)             # low BP in sicker patients

    on_oxygen = 0
    if code in RESPIRATORY_CODES and s == 2 and random.random() < 0.45:
        on_oxygen = 1
    if on_oxygen:
        spo2 = clamp(spo2, 93, 99)                # oxygen props up the reading

    consciousness = "Alert"
    if s == 2 and random.random() < 0.06:
        consciousness = random.choice(["Confusion", "Voice"])

    return {
        "respiratory_rate": int(round(clamp(rr, 8, 34))),
        "spo2": int(round(clamp(spo2, 82, 100))),
        "on_oxygen": on_oxygen,
        "systolic_bp": int(round(clamp(sbp, 80, 215))),
        "pulse": int(round(clamp(pulse, 46, 140))),
        "temperature_c": round(clamp(temp, 35.2, 40.0), 1),
        "consciousness": consciousness,
    }


def admission_outcome(age: int, severity: str, v: dict) -> str:
    """Admission chance rises with severity, age, and abnormal vitals."""
    s = SEV_NUM[severity]
    z = -3.0 + 1.5 * s + 0.03 * (age - 50)
    if v["spo2"] < 94:
        z += 1.0
    if v["respiratory_rate"] >= 24:
        z += 0.8
    if v["systolic_bp"] < 100 or v["systolic_bp"] >= 180:
        z += 0.8
    if v["pulse"] >= 111:
        z += 0.6
    if v["temperature_c"] >= 38.5:
        z += 0.4
    if v["consciousness"] != "Alert":
        z += 2.0
    probability = 1.0 / (1.0 + math.exp(-z))
    return "Admitted" if random.random() < probability else "Discharged"


def make_patient(index: int) -> dict:
    cond = random.choices(CONDITIONS, weights=[c["weight"] for c in CONDITIONS])[0]
    severity = random.choices(SEVERITIES, weights=cond["sev"])[0]

    age = int(round(clamp(random.gauss(*cond["age"]), 1, 100)))
    gender = random.choices(["Male", "Female", "Other"], weights=[0.48, 0.50, 0.02])[0]

    n_symptoms = random.randint(2, min(4, len(cond["symptoms"])))
    symptoms = random.sample(cond["symptoms"], n_symptoms)
    history = random.sample(cond["history"], random.randint(1, min(2, len(cond["history"]))))
    if "none" in history and len(history) > 1:
        history = [h for h in history if h != "none"]

    if not history:  # both picks were "none"
        history = ["none"]

    vitals = make_vitals(cond["code"], severity)
    row = {
        "patient_id": f"P{index:04d}",
        "age": age,
        "gender": gender,
        "symptoms": ";".join(symptoms),
        "medical_history": ";".join(history),
        **vitals,
        "diastolic_bp": int(round(clamp(vitals["systolic_bp"] * 0.62 + random.gauss(0, 5), 45, 130))),
        "disease_category": cond["category"],
        "icd10_code": cond["code"],
        "severity": severity,
        "admission_outcome": admission_outcome(age, severity, vitals),
    }
    return row


COLUMNS = [
    "patient_id", "age", "gender", "symptoms", "medical_history",
    "respiratory_rate", "spo2", "on_oxygen", "systolic_bp", "diastolic_bp",
    "pulse", "temperature_c", "consciousness",
    "disease_category", "icd10_code", "severity", "admission_outcome",
]
VITAL_COLUMNS = ["respiratory_rate", "spo2", "systolic_bp", "diastolic_bp", "pulse", "temperature_c"]


def inject_problems(rows: list[dict]) -> None:
    """Add the deliberate dirty data (missing values + a few outliers)."""
    # 1. Missing values: leave a vital-sign cell empty (None)
    for row in rows:
        for col in VITAL_COLUMNS:
            if random.random() < MISSING_RATE:
                row[col] = None

    # 2. Outliers: realistic data-entry mistakes on a few random rows
    picks = random.sample(range(len(rows)), 8)
    for i in picks[0:3]:
        rows[i]["temperature_c"] = round(random.uniform(97.5, 100.4), 1)   # Fahrenheit by mistake
    for i in picks[3:5]:
        rows[i]["pulse"] = random.choice([210, 240, 300])                  # extra digit
    for i in picks[5:7]:
        rows[i]["systolic_bp"] = random.choice([20, 260])                  # typo
    rows[picks[7]]["age"] = 150                                            # impossible age


def main() -> None:
    rows = [make_patient(i) for i in range(1, N_ROWS + 1)]
    inject_problems(rows)

    df = pd.DataFrame(rows, columns=COLUMNS)

    # "Int64" is pandas' integer type that allows empty cells, so the CSV
    # shows 98 instead of 98.0 and leaves missing cells truly empty.
    for col in ["age", "respiratory_rate", "spo2", "on_oxygen",
                "systolic_bp", "diastolic_bp", "pulse"]:
        df[col] = df[col].astype("Int64")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows x {len(df.columns)} columns -> {OUTPUT_PATH}")
    print("\nAdmission outcome counts:")
    print(df["admission_outcome"].value_counts().to_string())
    print("\nDisease category counts:")
    print(df["disease_category"].value_counts().to_string())
    print("\nSeverity counts:")
    print(df["severity"].value_counts().to_string())
    print("\nMissing values per column (only columns with any):")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())


if __name__ == "__main__":
    main()
