"""
deidentifier.py - removes personal information from patient data.

Owner: Member 1 (Data & Standards).

Why: hospital data is private. Before data is shared, exported, or used for
analysis, direct identifiers (names, phone numbers, ABHA/Aadhaar numbers,
e-mails, dates) must be hidden.

What it can do:
  1. deidentify_text()    - find and mask identifiers inside free text
  2. pseudonymize_id()    - turn a real ID into a stable fake ID (same input
                            -> same output, but it cannot be reversed)
  3. deidentify_record()  - clean a whole patient record (dict)

Limits (be honest about these in the demo):
  * It is rule-based (patterns), so it can miss unusual formats.
  * Names are only found when they follow a clue ("my name is ...",
    "Mr. ...") or are passed in via known_names. Names in Hindi script are
    only caught through known_names.
  * It is a privacy helper, NOT a guarantee of anonymity. A person must
    review before any real data is released.

Run the demo:
    python backend/data_governance/deidentifier.py
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Order matters: longer / more specific patterns first, so a 14-digit ABHA
# number is not mistaken for a 12-digit Aadhaar or a 10-digit phone number.
_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("ABHA_ID", re.compile(r"\b[\w.]+@(?:abdm|sbx)\b", re.IGNORECASE)),   # ABHA address
    ("ABHA_ID", re.compile(r"(?<!\d)\d{2}-?\d{4}-?\d{4}-?\d{4}(?!\d)")),      # 14-digit ABHA number
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),     # 12-digit
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
]

# Clues that a name follows. (?i:...) makes only the clue case-insensitive,
# while the name itself must start with a capital letter.
_NAME_PATTERNS = [
    re.compile(r"(?i:my name is|name is|name:|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"),
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"),
]

# Fields removed completely from a record (never kept, not even masked).
DROP_FIELDS = {"phone", "mobile", "email", "address", "abha_id", "aadhaar"}
# Fields that hold a person's name.
NAME_FIELDS = {"patient_name", "name"}
# Fields that hold an ID we want to keep linkable but fake.
ID_FIELDS = {"patient_id", "token"}


def pseudonymize_id(value: str, salt: Optional[str] = None) -> str:
    """Stable fake ID, e.g. 'P0001' -> 'PT-3f9a1c0b'.

    Uses HMAC-SHA256 with a secret salt. Set the secret in the environment
    variable MEDIKIOSK_DEID_SALT (never write it in code or commit it).
    """
    secret = salt or os.environ.get("MEDIKIOSK_DEID_SALT", "dev-only-salt-change-me")
    digest = hmac.new(secret.encode(), str(value).encode(), hashlib.sha256).hexdigest()
    return "PT-" + digest[:8]


def deidentify_text(text: str, known_names: Optional[Iterable[str]] = None) -> Tuple[str, List[str]]:
    """Mask identifiers in `text`.

    Returns (clean_text, findings) where findings lists the kinds found,
    e.g. ['PHONE', 'NAME'].
    """
    if not text:
        return text, []

    findings: List[str] = []
    clean = str(text)

    # 1. names we already know (e.g. the patient's name from check-in)
    for full in known_names or []:
        if not full or not str(full).strip():
            continue
        parts = [str(full).strip()] + [p for p in str(full).split() if len(p) >= 3]
        for part in sorted(set(parts), key=len, reverse=True):
            pattern = re.compile(r"(?<!\w)" + re.escape(part) + r"(?!\w)", re.IGNORECASE)
            clean, n = pattern.subn("[NAME]", clean)
            if n:
                findings.append("NAME")

    # 2. names after a clue ("my name is Ravi Kumar", "Dr. Sharma")
    for pat in _NAME_PATTERNS:
        def _mask_name(match: "re.Match[str]") -> str:
            findings.append("NAME")
            return match.group(0).replace(match.group(1), "[NAME]")
        clean = pat.sub(_mask_name, clean)

    # 3. numbers, e-mails, dates
    for label, pat in _PATTERNS:
        clean, n = pat.subn(f"[{label}]", clean)
        if n:
            findings.append(label)

    return clean, sorted(set(findings))


def deidentify_record(record: Dict[str, Any], known_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Return a cleaned copy of a patient record (the original is unchanged).

    * DROP_FIELDS are removed
    * NAME_FIELDS become '[NAME]'
    * ID_FIELDS become pseudonyms
    * every other text value is scanned with deidentify_text()
    """
    names = list(known_names or [])
    for key in NAME_FIELDS:
        if isinstance(record.get(key), str) and record[key].strip():
            names.append(record[key])

    def clean_value(value: Any) -> Any:
        if isinstance(value, str):
            return deidentify_text(value, names)[0]
        if isinstance(value, list):
            return [clean_value(v) for v in value]
        if isinstance(value, dict):
            return {k: clean_value(v) for k, v in value.items()}
        return value

    result: Dict[str, Any] = {}
    for key, value in record.items():
        lowered = key.lower()
        if lowered in DROP_FIELDS:
            continue
        if lowered in NAME_FIELDS:
            result[key] = "[NAME]"
        elif lowered in ID_FIELDS and value:
            result[key] = pseudonymize_id(str(value))
        else:
            result[key] = clean_value(value)
    return result


if __name__ == "__main__":
    print("=== Text example ===")
    sample = ("Hello, my name is Ravi Kumar. Call me on +91 98765 43210 or "
              "ravi.k@example.com. ABHA 12-3456-7890-1234, Aadhaar 1234 5678 9012. "
              "Seen by Dr. Sharma on 12/03/2024 for chest pain.")
    print("BEFORE:", sample)
    cleaned, found = deidentify_text(sample)
    print("AFTER: ", cleaned)
    print("FOUND: ", found)

    print("\n=== Record example ===")
    record = {
        "patient_id": "P0001",
        "patient_name": "Ravi Kumar",
        "phone": "9876543210",
        "age": 54,
        "chief_complaint": "Ravi has chest pain since 3 days",
        "notes": ["Daughter called 98765-43210"],
        "icd10_codes": [{"code": "I20.9", "name": "Angina pectoris"}],
    }
    print("BEFORE:", record)
    print("AFTER: ", deidentify_record(record))