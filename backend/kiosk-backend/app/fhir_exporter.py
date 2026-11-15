"""
fhir_exporter.py — HL7 / FHIR R4 Clinical Bundle Exporter for MediKiosk.
Conforms to HL7 FHIR Release 4 specification and Indian ABDM interoperability standards.
Supports both live Kiosk visit export and offline cohort export.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# fhir_exporter.py -> app -> kiosk-backend -> backend -> project root
ROOT = Path(__file__).resolve().parents[3]

# Fixed namespace UUID so the same patient ID generates deterministic resource UUIDs
_NAMESPACE = uuid.UUID("6f0d4f0e-5b1c-4c1e-9a52-2d1b6f2a0c11")

ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"     # WHO ICD-10
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
SNOMED = "http://snomed.info/sct"


def _load_module(name: str, path: Path):
    """Load a sibling module by file path."""
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_deid = _load_module("medikiosk_deidentifier", ROOT / "backend" / "data_governance" / "deidentifier.py")
_icd = _load_module("medikiosk_icd10_mapper", ROOT / "backend" / "converse-module" / "app" / "icd10_mapper.py")
ICD_NAMES: Dict[str, str] = {d["code"]: d["name"] for d in _icd.ICD10_DATABASE} if (_icd and hasattr(_icd, "ICD10_DATABASE")) else {}


def generate_fhir_bundle(visit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Live API endpoint exporter: transforms a MediKiosk visit record into
    a valid FHIR R4 Collection Bundle containing Patient, Condition (ICD-10),
    and Observation (NEWS2 + Vitals) resources.
    """
    token = visit_data.get("token", "UNKNOWN")
    now_iso = datetime.now(timezone.utc).isoformat()
    bundle_id = str(uuid.uuid4())
    patient_ref = f"urn:uuid:{uuid.uuid4()}"

    entries: List[Dict[str, Any]] = []

    # 1. FHIR Patient Resource
    patient_resource = {
        "fullUrl": patient_ref,
        "resource": {
            "resourceType": "Patient",
            "id": f"patient-{token}",
            "identifier": [
                {
                    "system": "https://medikiosk.abdm.gov.in/identifiers/patient",
                    "value": token
                }
            ],
            "active": True,
            "name": [
                {
                    "text": visit_data.get("patient_name") or f"Patient {token}"
                }
            ],
            "gender": (visit_data.get("gender") or "unknown").lower()
        }
    }
    entries.append(patient_resource)

    # 2. FHIR Condition Resource (ICD-10 Diagnostic Codes)
    history = visit_data.get("history", {})
    chief_complaint = history.get("chief_complaint", "")
    icd10_codes = history.get("icd10_codes", [])

    if not icd10_codes and chief_complaint:
        icd10_codes = [{"code": "R69", "name": "Illness, unspecified", "matched_keyword": "chief complaint"}]

    for idx, c in enumerate(icd10_codes):
        condition_resource = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Condition",
                "id": f"condition-{token}-{idx+1}",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active"
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis",
                                "display": "Encounter Diagnosis"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": c.get("code", "R69"),
                            "display": c.get("name", "Unspecified condition")
                        }
                    ],
                    "text": chief_complaint or c.get("name")
                },
                "subject": {
                    "reference": patient_ref
                },
                "recordedDate": now_iso
            }
        }
        entries.append(condition_resource)

    # 3. FHIR Observation Resource (NEWS2 Score & Vitals)
    vitals = visit_data.get("vitals", {})
    news2 = vitals.get("news2", {})
    news2_score = news2.get("total_score")

    if news2_score is not None:
        observation_resource = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "id": f"observation-news2-{token}",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "survey",
                                "display": "Survey"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "https://rcplondon.ac.uk/news2",
                            "code": "NEWS2",
                            "display": "National Early Warning Score 2"
                        }
                    ],
                    "text": "NEWS2 Composite Clinical Risk Score"
                },
                "subject": {
                    "reference": patient_ref
                },
                "effectiveDateTime": now_iso,
                "valueInteger": news2_score,
                "interpretation": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": (news2.get("risk_level") or "Low").upper(),
                                "display": f"{news2.get('risk_level', 'Low')} Risk"
                            }
                        ]
                    }
                ],
                "component": [
                    {
                        "code": {"text": "Systolic Blood Pressure"},
                        "valueQuantity": {"value": vitals.get("systolic_bp"), "unit": "mmHg"}
                    },
                    {
                        "code": {"text": "Heart Rate / Pulse"},
                        "valueQuantity": {"value": vitals.get("pulse"), "unit": "beats/min"}
                    },
                    {
                        "code": {"text": "SpO2 (Oxygen Saturation)"},
                        "valueQuantity": {"value": vitals.get("spo2"), "unit": "%"}
                    },
                    {
                        "code": {"text": "Respiration Rate"},
                        "valueQuantity": {"value": vitals.get("respiratory_rate"), "unit": "breaths/min"}
                    },
                    {
                        "code": {"text": "Body Temperature"},
                        "valueQuantity": {"value": vitals.get("temperature_c"), "unit": "Cel"}
                    }
                ]
            }
        }
        entries.append(observation_resource)

    # Composite Bundle
    bundle = {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": now_iso,
        "total": len(entries),
        "entry": entries
    }
    return bundle


# ==============================================================================
# Cohort & Standards Exporter (Member 1 / ABDM Governance Integration)
# ==============================================================================

_SIMPLE_VITALS = [
    ("respiratory_rate", "9279-1", "Respiratory rate", "/min", "/min"),
    ("spo2", "59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry", "%", "%"),
    ("pulse", "8867-4", "Heart rate", "/min", "/min"),
    ("temperature_c", "8310-5", "Body temperature", "\u00b0C", "Cel"),
]

_SEVERITY_SNOMED = {
    "mild": ("255604002", "Mild"),
    "moderate": ("6736007", "Moderate"),
    "severe": ("24484000", "Severe"),
}
_GENDER = {"male": "male", "female": "female", "other": "other"}


def _uid(*parts: Any) -> str:
    return str(uuid.uuid5(_NAMESPACE, "|".join(str(p) for p in parts)))


def _number(value: Any) -> Optional[float]:
    """Return a float, or None if the value is empty / not a number."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _tidy(number: float) -> Any:
    return int(number) if float(number).is_integer() else number


def _observation(obs_id: str, patient_ref: str, enc_ref: str, when: str, loinc_code: str,
                 display: str, vital_sign: bool = True, quantity: Optional[dict] = None,
                 components: Optional[List[dict]] = None) -> dict:
    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "code": {"coding": [{"system": LOINC, "code": loinc_code, "display": display}], "text": display},
        "subject": {"reference": patient_ref},
        "encounter": {"reference": enc_ref},
        "effectiveDateTime": when,
    }
    if vital_sign:
        obs["category"] = [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "vital-signs", "display": "Vital Signs"}]}]
    if quantity is not None:
        obs["valueQuantity"] = quantity
    if components:
        obs["component"] = components
    return obs


def _quantity(value: float, unit: str, ucum_code: str) -> dict:
    return {"value": _tidy(value), "unit": unit, "system": UCUM, "code": ucum_code}


def build_fhir_bundle(record: Dict[str, Any], deidentify: bool = True) -> Dict[str, Any]:
    """Build a FHIR R4 Bundle from one visit record.

    record keys (all optional except patient_id):
        patient_id, patient_name, age, gender, chief_complaint, severity,
        icd10_codes: [{"code": "I10", "name": "..."}],
        vitals: {respiratory_rate, spo2, systolic_bp, diastolic_bp, pulse, temperature_c}
    """
    if "patient_id" not in record:
        raise ValueError("record needs a 'patient_id'")
    if deidentify and _deid is None:
        # Fallback if deidentifier is missing
        shown_id = f"P-ANON-{record['patient_id']}"
    else:
        real_id = str(record["patient_id"])
        shown_id = _deid.pseudonymize_id(real_id) if (deidentify and _deid) else real_id

    real_id = str(record["patient_id"])
    patient_uid = _uid("patient", real_id)
    enc_uid = _uid("encounter", real_id)
    patient_ref, enc_ref = f"urn:uuid:{patient_uid}", f"urn:uuid:{enc_uid}"
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # --- Patient ---
    patient: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_uid,
        "identifier": [{"system": "https://medikiosk.example/patient-id", "value": shown_id}],
        "gender": _GENDER.get(str(record.get("gender", "")).lower(), "unknown"),
    }
    if not deidentify and record.get("patient_name"):
        patient["name"] = [{"text": str(record["patient_name"])}]

    # --- Encounter ---
    encounter: Dict[str, Any] = {
        "resourceType": "Encounter",
        "id": enc_uid,
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                  "code": "AMB", "display": "ambulatory"},
        "subject": {"reference": patient_ref},
    }
    complaint = record.get("chief_complaint")
    if complaint:
        if deidentify and _deid:
            names = [record["patient_name"]] if record.get("patient_name") else []
            complaint = _deid.deidentify_text(str(complaint), names)[0]
        encounter["reasonCode"] = [{"text": str(complaint)}]

    entries: List[Dict[str, Any]] = [patient, encounter]

    # --- Conditions (one per ICD-10 code) ---
    severity = str(record.get("severity", "")).lower()
    for item in record.get("icd10_codes") or []:
        code = item.get("code")
        if not code:
            continue
        name = item.get("name") or ICD_NAMES.get(code, code)
        condition: Dict[str, Any] = {
            "resourceType": "Condition",
            "id": _uid("condition", real_id, code),
            "clinicalStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "provisional"}]},
            "code": {"coding": [{"system": ICD10_SYSTEM, "code": code, "display": name}], "text": name},
            "subject": {"reference": patient_ref},
            "encounter": {"reference": enc_ref},
            "recordedDate": now,
        }
        if severity in _SEVERITY_SNOMED:
            sct_code, sct_display = _SEVERITY_SNOMED[severity]
            condition["severity"] = {"coding": [{"system": SNOMED, "code": sct_code, "display": sct_display}]}
        entries.append(condition)

    # --- Observations: age, vitals ---
    age = _number(record.get("age"))
    if age is not None:
        entries.append(_observation(_uid("obs", real_id, "age"), patient_ref, enc_ref, now,
                                    "30525-0", "Age", vital_sign=False,
                                    quantity=_quantity(age, "years", "a")))

    vitals = record.get("vitals") or {}
    for field, loinc_code, display, unit, ucum_code in _SIMPLE_VITALS:
        value = _number(vitals.get(field))
        if value is not None:
            entries.append(_observation(_uid("obs", real_id, field), patient_ref, enc_ref, now,
                                        loinc_code, display, quantity=_quantity(value, unit, ucum_code)))

    systolic, diastolic = _number(vitals.get("systolic_bp")), _number(vitals.get("diastolic_bp"))
    components = []
    if systolic is not None:
        components.append({"code": {"coding": [{"system": LOINC, "code": "8480-6", "display": "Systolic blood pressure"}]},
                           "valueQuantity": _quantity(systolic, "mmHg", "mm[Hg]")})
    if diastolic is not None:
        components.append({"code": {"coding": [{"system": LOINC, "code": "8462-4", "display": "Diastolic blood pressure"}]},
                           "valueQuantity": _quantity(diastolic, "mmHg", "mm[Hg]")})
    if components:
        entries.append(_observation(_uid("obs", real_id, "bp"), patient_ref, enc_ref, now,
                                    "85354-9", "Blood pressure panel", components=components))

    return {
        "resourceType": "Bundle",
        "id": _uid("bundle", real_id),
        "meta": {"tag": [{"system": "https://medikiosk.example/tags", "code": "synthetic-test-data",
                          "display": "Synthetic data - not a real patient"}]},
        "type": "collection",
        "timestamp": now,
        "entry": [{"fullUrl": f"urn:uuid:{res['id']}", "resource": res} for res in entries],
    }


def validate_bundle(bundle: Dict[str, Any]) -> List[str]:
    """Basic self-check. Returns a list of problems (empty list = looks fine)."""
    problems: List[str] = []
    if bundle.get("resourceType") != "Bundle":
        problems.append("resourceType must be 'Bundle'")
    if bundle.get("type") not in {"collection", "document", "transaction", "batch"}:
        problems.append("Bundle.type missing or invalid")

    entries = bundle.get("entry") or []
    if not entries:
        problems.append("Bundle has no entries")
    urls = {e.get("fullUrl") for e in entries}
    kinds: Dict[str, int] = {}

    for e in entries:
        res = e.get("resource", {})
        kind = res.get("resourceType")
        kinds[kind] = kinds.get(kind, 0) + 1
        if not kind or not res.get("id"):
            problems.append("an entry has no resourceType or id")
        if e.get("fullUrl") != f"urn:uuid:{res.get('id')}":
            problems.append(f"{kind}: fullUrl does not match id")
        for key in ("subject", "encounter"):
            ref = res.get(key, {}).get("reference") if isinstance(res.get(key), dict) else None
            if ref and ref not in urls:
                problems.append(f"{kind}.{key} points to a missing resource: {ref}")
        if kind == "Condition":
            coding = (res.get("code", {}).get("coding") or [{}])[0]
            if not coding.get("system") or not coding.get("code"):
                problems.append("Condition needs code.coding.system and code")
        if kind == "Observation" and "valueQuantity" not in res and "component" not in res:
            problems.append("Observation has no value")

    if kinds.get("Patient", 0) != 1:
        problems.append("Bundle must contain exactly one Patient")
    return problems


def row_to_record(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert one row of the cohort CSV into the record build_fhir_bundle expects."""
    code = row.get("icd10_code", "")
    return {
        "patient_id": row["patient_id"],
        "age": row.get("age"),
        "gender": row.get("gender"),
        "chief_complaint": (row.get("symptoms") or "").replace(";", ", "),
        "severity": row.get("severity"),
        "icd10_codes": [{"code": code, "name": ICD_NAMES.get(code, code)}] if code else [],
        "vitals": {k: row.get(k) for k in ("respiratory_rate", "spo2", "systolic_bp",
                                           "diastolic_bp", "pulse", "temperature_c")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export synthetic patients as FHIR R4 Bundles")
    parser.add_argument("--csv", default=str(ROOT / "data" / "synthetic_opd_cohort_500_clean.csv"))
    parser.add_argument("--patients", default="P0001,P0002,P0003", help="comma-separated patient IDs")
    parser.add_argument("--out", default=str(ROOT / "data" / "fhir_samples"))
    args = parser.parse_args()

    wanted = [p.strip() for p in args.patients.split(",") if p.strip()]
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = {r["patient_id"]: r for r in csv.DictReader(fh)}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in wanted:
        if pid not in rows:
            print(f"{pid}: not found in the CSV")
            continue
        bundle = build_fhir_bundle(row_to_record(rows[pid]), deidentify=True)
        problems = validate_bundle(bundle)
        path = out_dir / f"{pid}_bundle.json"
        path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        counts: Dict[str, int] = {}
        for e in bundle["entry"]:
            kind = e["resource"]["resourceType"]
            counts[kind] = counts.get(kind, 0) + 1
        print(f"{pid}: wrote {path.name} | {counts} | check: {'OK' if not problems else problems}")


if __name__ == "__main__":
    main()
