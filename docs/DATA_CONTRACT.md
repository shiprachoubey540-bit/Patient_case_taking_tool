# MediKiosk Data Contract (v1 DRAFT)

Written by Member 1. Members 2 and 3 must confirm before we rely on it.
To change anything, tell the whole team first, then update this file.

All data is SYNTHETIC. It is never real clinical evidence.

## 1. File paths

| What | Path | Owner |
|---|---|---|
| Raw synthetic cohort | `data/synthetic_opd_cohort_500.csv` | Member 1 |
| Cleaned cohort (for ML) | `data/synthetic_opd_cohort_500_clean.csv` | Member 1 |
| Data dictionary | `docs/DATA_DICTIONARY.md` | Member 1 |
| De-identifier | `backend/data_governance/deidentifier.py` | Member 1 |
| ICD-10 mapper | `backend/converse-module/app/icd10_mapper.py` | Member 1 |
| FHIR exporter | `backend/kiosk-backend/app/fhir_exporter.py` | Member 1 |
| NEWS2 engine | `backend/kiosk-backend/app/news2_engine.py` | Member 2 |
| CDS explainer | `backend/kiosk-backend/app/cds_explainer.py` | Member 2 |
| Trained models | `models/` | Member 2 |
| Notebooks | `notebooks/` | each owner |

## 2. CSV columns (Member 1 -> Member 2)

All column names are lowercase snake_case. One row = one outpatient visit.

| Column | Type | Allowed values / range | Notes |
|---|---|---|---|
| `patient_id` | text | `P0001` ... `P0500` | Fake ID, never a real one |
| `age` | integer | 1 to 100 | years |
| `gender` | text | `Male`, `Female`, `Other` | |
| `symptoms` | text | semicolon-separated, e.g. `chest pain;sweating` | |
| `medical_history` | text | semicolon-separated, or `none` | |
| `respiratory_rate` | integer | breaths/min | NEWS2 input |
| `spo2` | integer | percent | NEWS2 input |
| `on_oxygen` | integer | `0` or `1` | NEWS2 input (1 = supplemental oxygen) |
| `systolic_bp` | integer | mmHg | NEWS2 input |
| `diastolic_bp` | integer | mmHg | |
| `pulse` | integer | beats/min | NEWS2 input |
| `temperature_c` | float | degrees Celsius | NEWS2 input |
| `consciousness` | text | `Alert`, `Confusion`, `Voice`, `Pain`, `Unresponsive` | NEWS2 input |
| `disease_category` | text | `Circulatory`, `Respiratory`, `Digestive`, `Endocrine_Metabolic`, `Musculoskeletal` | |
| `icd10_code` | text | e.g. `I10` | |
| `severity` | text | `Mild`, `Moderate`, `Severe` | |
| `admission_outcome` | text | `Discharged`, `Admitted` | ML target |

Missing values are left EMPTY in the raw CSV (not 0, not "NA").

## 3. JSON shapes

NEWS2 result (Member 2 -> Member 3):
```json
{"score": 5, "level": "Amber"}
```
`score` is an integer. `level` is exactly `Green`, `Amber` or `Red`.

Top 3 risk factors (Member 2 -> Member 3):
```json
[{"factor": "age", "impact": 0.31}]
```
A list of exactly 3 items, `factor` = a CSV column name, `impact` = a number.

ML prediction (Member 2 -> Member 3):
```json
{"label": "Admitted", "probability": 0.72}
```
This is a statistical estimate, NOT a diagnosis.

ICD-10 codes (Member 1 -> Member 3), already used by the dashboard:
```json
[{"code": "I10", "name": "Essential (primary) hypertension", "matched_keyword": "high blood pressure"}]
```

## 4. Open question

The kiosk does not collect vital signs yet. Assumption for now: a nurse
enters them on the doctor page. Team to confirm.