"""
Local backend — uses Ollama with a vision model (e.g. llama3.2-vision) to 
extract structured data from medical documents.
"""
import base64
import json
import os
import httpx
from typing import Optional

from .prompts import build_extraction_instruction
from .schema import ExtractedDocument

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")

def is_available() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False

def extract(image_bytes: bytes, mime_type: str, document_type_hint: Optional[str] = None) -> ExtractedDocument:
    if not is_available():
        raise RuntimeError(f"Ollama is unreachable at {OLLAMA_URL}")

    instruction = build_extraction_instruction(document_type_hint)
    b64_img = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": instruction + "\n\nReturn ONLY valid JSON matching the exact schema requested.",
                "images": [b64_img]
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1500
        }
    }

    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=90.0)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to communicate with Ollama: {exc}") from exc

    body = resp.json()
    content = body.get("message", {}).get("content", "")
    
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return ExtractedDocument.model_validate_json(content)
    except Exception as exc:
        print(f"[ocr/ollama] JSON parsing failed. Raw output:\n{content}")
        raise RuntimeError(f"Ollama returned invalid JSON: {exc}") from exc
