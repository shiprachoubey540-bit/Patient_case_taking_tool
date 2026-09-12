"""
Real backend — Gemini's multimodal input, same Interactions API and
structured-output pattern as the Converse module's app/llm.py.

Verified against https://ai.google.dev/gemini-api/docs (Sept 2026): image
input goes in the `input` list as a part with type "image", base64-encoded
data, and a mime_type, alongside a text part. If Google has changed this
interface again by the time you read this, check that page before assuming
this code is wrong.
"""
from __future__ import annotations

import base64
import os

from google import genai

from .prompts import build_extraction_instruction
from .schema import ExtractedDocument

DEFAULT_MODEL = "gemini-3.5-flash"


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def extract(image_bytes: bytes, mime_type: str, document_type_hint: str | None = None) -> ExtractedDocument:
    client = genai.Client()  # reads GEMINI_API_KEY from the environment

    instruction = build_extraction_instruction(document_type_hint)

    interaction = client.interactions.create(
        model=_model_name(),
        input=[
            {"type": "text", "text": instruction},
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
                "mime_type": mime_type,
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ExtractedDocument.model_json_schema(),
        },
    )

    return ExtractedDocument.model_validate_json(interaction.output_text)
