"""
No-network stand-in for whisper_backend.py — same purpose as the other
modules' mock backends: exercise the upload/validation/plumbing without
needing faster-whisper installed or a model downloaded.

Doesn't actually listen to the audio — returns a fixed canned transcript
regardless of what's in the file, so tests and offline development can
exercise the rest of the pipeline deterministically.
"""
from __future__ import annotations

from typing import Optional


def transcribe(audio_bytes: bytes, language: Optional[str] = None) -> dict:
    return {
        "text": "(mock transcript) I have a headache since yesterday afternoon",
        "language_detected": language or "en",
        "language_probability": 1.0,
    }
