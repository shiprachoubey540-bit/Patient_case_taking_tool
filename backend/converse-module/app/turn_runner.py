"""
Wraps a backend's run_turn() to handle "empty" turns: schema-valid, but with
no next_question and not marked complete either — i.e. nothing for the UI to
show and nothing for the interview to do next.

This showed up in practice with the local Ollama backend (llama3.1:8b), and
consistently — not a one-off. Ollama's `format` parameter enforces the JSON
*shape*, not that the model fills the fields sensibly, and a smaller/local
model is more likely to leave next_question null without setting
complete=true than Gemini is.

Behavior: retry once, and if it's still empty, fall back to a generic
continuation question and keep the interview moving rather than stopping it
cold. The point of the Ollama backend is to keep a demo alive if venue wifi
dies — a hard failure on a known, somewhat common model quirk defeats that
purpose. A real exception (network error, invalid JSON that fails schema
validation) is NOT swallowed here — it propagates so main.py can still
surface a clear 502; this only papers over "technically valid, semantically
empty" responses.

2026-09-12 fix: the retry used to just call the backend again with the
IDENTICAL transcript and system prompt. Confirmed on real hardware (legion)
that this often does NOT help — gemma4:12b left next_question blank "twice
in a row" on the very first interview turn, not as a rare coincidence. At
temperature 0.3 with nothing else different, a local model has very little
reason to answer differently the second time. The retry now passes
prompts.py's RETRY_HINT into the backend's system prompt, naming the exact
mistake and re-stating what's required, so the retry has an actual reason to
diverge instead of relying on sampling noise alone.

If you'd rather know immediately every time this happens instead of getting
a generic fallback question, that's a reasonable choice too — swap the
fallback below for `raise RuntimeError(...)`.
"""
from __future__ import annotations

import sys
from typing import Callable, List, Optional

from .prompts import RETRY_HINT
from .schema import TurnResult

TurnFn = Callable[[List[dict], bool, str, int, Optional[str], Optional[str]], TurnResult]

# Localized so a patient mid-Hindi-interview doesn't suddenly get an English
# sentence at exactly the moment the backend is already struggling — see
# prompts.py's module docstring for the same "generated, not reviewed" caveat.
FALLBACK_QUESTIONS = {
    "en": "Sorry, could you tell me a bit more about that?",
    "hi": "माफ़ कीजिए, क्या आप इसके बारे में थोड़ा और बता सकते हैं?",
}


def _is_empty_turn(result: TurnResult) -> bool:
    return not result.complete and not result.next_question


def _last_patient_text(transcript: List[dict]) -> str:
    return transcript[-1]["text"] if transcript else "(no transcript)"


def run_turn_with_retry(
    turn_fn: TurnFn, transcript: List[dict], ayush_mode: bool, language: str, turn_count: int, current_history: Optional[str] = None
) -> TurnResult:
    result = turn_fn(transcript, ayush_mode, language, turn_count, None, current_history)
    if _is_empty_turn(result):
        print(
            f"[turn_runner] empty turn #1 — last patient text: {_last_patient_text(transcript)!r} | "
            f"acknowledgement={result.acknowledgement!r} | history_so_far={result.history.model_dump_json()}",
            file=sys.stderr,
        )
        # The retry used to repeat this exact call with nothing changed —
        # at low-but-nonzero temperature that gave a local model very
        # little reason to answer differently, and in practice it often
        # didn't (see prompts.py's RETRY_HINT docstring: this was observed
        # failing "twice in a row" on live hardware, not as a rare fluke).
        # Passing RETRY_HINT here makes this call meaningfully different
        # from the first one instead of a coin-flip re-roll.
        result = turn_fn(transcript, ayush_mode, language, turn_count, RETRY_HINT, current_history)
        if _is_empty_turn(result):
            print(
                "[turn_runner] backend returned an empty turn twice in a row "
                f"(no next_question, not complete) — falling back to a generic question | "
                f"last patient text: {_last_patient_text(transcript)!r} | acknowledgement={result.acknowledgement!r} | "
                f"history_so_far={result.history.model_dump_json()}",
                file=sys.stderr,
            )
            fallback = FALLBACK_QUESTIONS.get(language, FALLBACK_QUESTIONS["en"])
            result = result.model_copy(update={"next_question": fallback})
    return result
