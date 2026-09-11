"""
Merges Module A's conversational history with Module B's digitized
document data into one structured case summary.

Module A's real output format isn't finalized yet, so this expects a
reasonable placeholder shape:
    {
      "chief_complaint": "...",
      "symptoms": ["...", "..."],
      "duration": "...",
      "reported_medications": ["...", "..."]   (optional, patient-reported)
    }
Once your teammate finalizes Module A's actual schema, update
`normalize_conversation_history()` below to match it — that's the only
function that needs to change.
"""


def normalize_conversation_history(raw: dict) -> dict:
    """Adapts Module A's raw output into the shape this merger expects.
    Update this if Module A's real schema differs from the placeholder."""
    return {
        "chief_complaint": raw.get("chief_complaint", ""),
        "symptoms": raw.get("symptoms", []),
        "duration": raw.get("duration", ""),
        "reported_medications": raw.get("reported_medications", []),
    }


def merge_case_summary(conversation_history: dict, document_data: dict) -> dict:
    """
    conversation_history: raw dict from Module A
    document_data: the JSON Module B's /digitize endpoint returns
    """
    convo = normalize_conversation_history(conversation_history)
    structured = document_data.get("structured_data", {})

    # Merge medicines: document-extracted ones are more reliable than
    # patient-reported ones, so keep both but tag their source.
    medicines = []
    for med in structured.get("medicines", []):
        medicines.append({**med, "source": "document"})
    for med in convo.get("reported_medications", []):
        medicines.append({"raw_line": med, "dosage": None, "frequency": None,
                           "source": "patient_reported"})

    return {
        "chief_complaint": convo["chief_complaint"],
        "reported_symptoms": convo["symptoms"],
        "duration": convo["duration"],
        "diagnoses": structured.get("diagnoses", []),
        "medicines": medicines,
        "relevant_dates": structured.get("dates", []),
        "lab_values": structured.get("lab_values", []),
        "document_ocr_confidence": document_data.get("ocr_confidence"),
        "needs_human_review": document_data.get("needs_human_review", False),
        "source_document": document_data.get("source_filename"),
    }
