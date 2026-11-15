"""
fhir_exporter.py — HL7 / FHIR R4 Clinical Bundle Exporter for MediKiosk.
Conforms to HL7 FHIR Release 4 specification and Indian ABDM interoperability standards.
Maps Kiosk intake, ICD-10 diagnoses, and NEWS2 vital scores into standard FHIR resources.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List


def generate_fhir_bundle(visit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms a MediKiosk visit record into a valid FHIR R4 Collection Bundle.
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

    # If icd10_codes not pre-computed, provide fallback
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
