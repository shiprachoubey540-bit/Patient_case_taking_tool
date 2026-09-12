"""
Real LLM backend, using the Gemini API's Interactions endpoint with
structured JSON output.

Verified against https://ai.google.dev/gemini-api/docs (Sept 2026) — the
SDK is `google-genai`, calls go through `client.interactions.create(...)`,
and `genai.Client()` picks up the GEMINI_API_KEY environment variable
automatically. If Google has changed this interface again by the time you
read this, check that page first before assuming this code is wrong.

Design choice: we deliberately do NOT use `previous_interaction_id` for
multi-turn state. Instead we resend the rendered transcript as `input` each
turn and ask the model to re-derive the full ClinicalHistory from scratch.
That keeps our own session store as the single source of truth for
conversation state (important for an eventual audit trail on clinical data)
and sidesteps any question of whether structured-output schemas persist
correctly across chained interactions.

Latency fix (2026-09-07): live testing on legion showed real per-turn waits
of 30-60s+ by conversation turn 8, well above the ~14-15s baseline seen
early in testing. Root cause, confirmed against Gemini's docs rather than
guessed at: this call was passing no `generation_config` at all, and
gemini-3.5-flash defaults to thinking_level="medium" when none is given —
so every turn paid for a full extended-reasoning pass before emitting the
structured JSON, stacking on top of the cost that already grows with
transcript length (see prompts.py/BASE_INSTRUCTION's "re-derive the FULL
history every turn, from the entire transcript" design, and today's three
stacked prompt additions — medications split, triage_level, language
instruction — which also made the system prompt itself longer). This task
is close to pure structured extraction with a little judgment layered in
(red_flag, triage_level) — not the multi-step agentic reasoning "medium"/
"high" thinking is meant for — so lowering thinking level is the correct
fix, not a workaround for a symptom.

GEMINI_THINKING_LEVEL defaults to "low" rather than "minimal": "minimal" is
documented as tuned for quick factual chat/simple tool calls, and this call
still needs the model to weigh red_flag/triage_level judgment calls, not
just restate facts back — going straight to the fastest setting risked
trading a real safety-relevant judgment for speed without evidence it's
still accurate. If per-turn latency is still too high after this change,
"minimal" is the next thing to try, via .env (GEMINI_THINKING_LEVEL=minimal)
with no code change — but that trade-off should be a deliberate call made
after watching real triage_level/red_flag quality at "low" in practice, not
a default baked in here without seeing it hold up.
"""
from __future__ import annotations

import os
import sys
import time

from google import genai

from .prompts import build_system_instruction, render_transcript
from .schema import TurnResult

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_THINKING_LEVEL = "low"


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _thinking_level() -> str:
    return os.environ.get("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL)


def run_turn(
    transcript: list[dict],
    ayush_mode: bool = False,
    language: str = "en",
    turn_count: int = 1,
    retry_hint: str | None = None,
    current_history: str | None = None,
) -> TurnResult:
    """
    transcript: list of {"role": "assistant"|"patient", "text": str}, already
    including the patient's latest message.

    retry_hint: only set by turn_runner.py's retry, after a first empty
    turn — see prompts.py's RETRY_HINT for why. Gemini rarely needs this
    (it's a local-model quirk mostly), but the signature stays uniform
    across all three backends so turn_runner can call any of them the same
    way.
    """
    client = genai.Client()  # reads GEMINI_API_KEY from the environment

    system_instruction = build_system_instruction(ayush_mode, language, retry_hint, current_history=current_history)
    input_text = render_transcript(transcript)
    thinking_level = _thinking_level()

    started = time.monotonic()
    interaction = client.interactions.create(
        model=_model_name(),
        system_instruction=system_instruction,
        input=input_text,
        generation_config={"thinking_level": thinking_level},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": TurnResult.model_json_schema(),
        },
    )
    elapsed = time.monotonic() - started
    # Printed every turn, not just when slow, so real latency trends across
    # a whole interview show up in the server log instead of only being
    # noticed from a screenshot after the fact (see 2026-09-07 note above).
    print(
        f"[llm/gemini] turn_count={turn_count} thinking_level={thinking_level!r} "
        f"transcript_chars={len(input_text)} elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )

    # from_raw_lenient, not plain model_validate_json — see schema.py's
    # docstring for why (next_question's min_length=1 fix, 2026-09-12).
    return TurnResult.from_raw_lenient(interaction.output_text)
