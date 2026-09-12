"""
In-memory document store — same "deliberately simple for a hackathon
prototype" approach as the Converse module's session_store.py. Swap for a
real database later without touching the request-handling code, if needed.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from dateutil import parser as dateparser

from .schema import ExtractedDocument, StoredDocument

_DOCS: Dict[str, StoredDocument] = {}


def save(extracted: ExtractedDocument, backend: str, filename: Optional[str], session_id: Optional[str]) -> StoredDocument:
    doc = StoredDocument(
        id=str(uuid.uuid4()),
        session_id=session_id,
        filename=filename,
        uploaded_at=time.time(),
        backend=backend,
        extracted=extracted,
    )
    _DOCS[doc.id] = doc
    return doc


def get(doc_id: str) -> StoredDocument:
    if doc_id not in _DOCS:
        raise KeyError(doc_id)
    return _DOCS[doc_id]


def list_all(session_id: Optional[str] = None) -> List[StoredDocument]:
    docs = list(_DOCS.values())
    if session_id is not None:
        docs = [d for d in docs if d.session_id == session_id]
    return docs


def parse_document_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Best-effort parse of the free-text date the model read off the document.
    Deliberately permissive (dayfirst, fuzzy — Indian documents are usually
    DD/MM/YYYY and often have surrounding text like "Date:"), and
    deliberately allowed to fail: a document_date the model transcribed
    faithfully but that doesn't parse is not an error, just un-sortable.
    """
    if not date_str:
        return None
    try:
        return dateparser.parse(date_str, dayfirst=True, fuzzy=True)
    except (ValueError, OverflowError):
        return None


def timeline(session_id: Optional[str] = None) -> List[dict]:
    """
    Documents ordered chronologically where a date could be parsed off the
    document itself, falling back to upload order for the rest — appended
    after the dated ones, not interleaved by guesswork. Each entry says
    which case it is (`date_source`) so the UI doesn't overstate certainty.
    """
    docs = list_all(session_id)

    dated = []
    undated = []
    for doc in docs:
        parsed = parse_document_date(doc.extracted.document_date)
        if parsed:
            dated.append((parsed, doc))
        else:
            undated.append(doc)

    dated.sort(key=lambda pair: pair[0])
    undated.sort(key=lambda d: d.uploaded_at)

    entries = [
        {"document": doc, "date_source": "document_date", "sort_date": parsed.isoformat()}
        for parsed, doc in dated
    ] + [
        {"document": doc, "date_source": "upload_order", "sort_date": None}
        for doc in undated
    ]
    return entries
