"""
fhir_exporter.py - turns one patient visit into a FHIR R4 JSON Bundle.

Owner: Member 1 (Data & Standards).

What is FHIR? A standard format that lets hospital systems share health data.
A "Bundle" is one JSON file that holds several small records ("resources"):

    Patient      who (de-identified: pseudonym ID, no name)
    Encounter    the outpatient visit and the reason for it
    Condition    each ICD-10 code (status: PROVISIONAL, see below)
    Observation  age and each vital sign (LOINC-coded, UCUM units)

Safety notes
  * Conditions are marked verificationStatus = "provisional": they come from
    keyword matching, NOT from a doctor's diagnosis.
  * By default the export is de-identified (see backend/data_governance/
    deidentifier.py). The patient's name is never written.
  * Synthetic data is tagged "synthetic-test-data".
  * This file builds valid-looking FHIR R4, and a small built-in checker
    (validate_bundle) tests the basics. For a full check, paste a sample into
    https://validator.fhir.org

Run the demo (from the project root):
    python backend/kiosk-backend/app/fhir_exporter.py
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

# Fixed value so the same patient always gets the same resource IDs.
_NAMESPACE = uuid.UUID("6f0d4f0e-5b1c-4c1e-9a52-2d1b6f2a0c11")

ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"     # WHO ICD-10
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
SNOMED = "http://snomed.info/sct"


def _load_module(name: str, path: Path):
    """Load a sibling module by file path (avoids clashes between the
    several packages that are all called 'app')."""
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_deid = _load_module("medikiosk_deidentifier", ROOT / "backend" / "data_governance" / "deidentifier.py")
_icd = _load_module("medikiosk_icd10_mapper", ROOT / "backend" / "converse-module" / "app" / "icd10_mapper.py")
ICD_NAMES: Dict[str, str] = {d["code"]: d["name"] for d in _icd.ICD10_DATABASE} if _icd else {}

# (record field, LOINC code, display, unit text, UCUM code)
_SIMPLE_VITALS = [
    ("respiratory_rate", "9279-1", "Respiratory rate", "/min", "/min"),
    ("spo2", "59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry", "%", "%"),
    ("pulse", "8867-4", "Heart rate", "/min", "/min"),
    ("temperature_c", "8310-5", "Body temperature", "\u00b0C", "Cel"),
]

# FHIR Condition.severity uses SNOMED CT codes
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
        raise RuntimeError("backend/data_governance/deidentifier.py not found - cannot de-identify")

    real_id = str(record["patient_id"])
    shown_id = _deid.pseudonymize_id(real_id) if deidentify else real_id
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
        if deidentify:
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