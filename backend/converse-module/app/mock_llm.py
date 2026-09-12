"""
Deterministic, no-network stand-in for llm.py.

Used automatically when GEMINI_API_KEY isn't set, so:
- the rest of the team can build/demo against this module before everyone
  has their own API key,
- tests run in CI / offline without hitting a real API or burning quota,
- you always have a fallback if the live API is flaky right before a demo.

It doesn't understand the patient's actual words — it just marches through a
fixed question script and records whatever the patient typed into the
matching field, positionally. Good enough to prove the plumbing works; not a
substitute for testing against the real model before you rely on it.

Multilingual note (added 2026-09-07): this backend now asks its scripted
questions in the session's language (currently "en" or "hi", see
prompts.py's LANGUAGE_NAMES), so the mic/chat UI can be exercised end-to-end
in Hindi without needing a live Gemini/Ollama call. But unlike the real
backends, it does NOT translate the patient's answer into English for the
structured record — real backends are instructed to always record history
fields in English regardless of interview language (see
prompts.py::_language_instruction); this mock has no language understanding
at all, so it just stores whatever text the patient typed, verbatim, in
whatever language that was. If you're testing the "record stays in English"
behavior specifically, that only actually happens against gemini/ollama, not
mock — this is a known, deliberate limitation of a script-replay stand-in,
not a bug.
"""
from __future__ import annotations

from .schema import AyushAssessment, ClinicalHistory, HPI, TurnResult

_CORE_SCRIPT = [
    ("chief_complaint", "What's the main problem that brought you in today?"),
    ("hpi.site", "Where exactly do you feel it?"),
    ("hpi.onset", "When did this start?"),
    ("hpi.character", "How would you describe the feeling — sharp, dull, burning, something else?"),
    ("hpi.severity", "On a scale of 1 to 10, how bad is it?"),
    ("past_medical_surgical_history", "Have you had any major illnesses or surgeries before?"),
    ("current_medications", "Are you currently taking any medicines?"),
    ("drug_allergy_history", "Are you allergic to any medicines?"),
    ("family_history", "Does anyone in your immediate family have a similar or related condition?"),
]

# Same fields, same order as _CORE_SCRIPT — kept as a parallel list (not a
# dict keyed by field) so it's visually easy to eyeball that nothing's out
# of sync. Generated, not reviewed by a native/clinical Hindi speaker — see
# prompts.py's module docstring for the same caveat.
_CORE_QUESTIONS_HI = [
    "आज आप किस समस्या के लिए यहाँ आए हैं?",
    "आपको ठीक-ठीक कहाँ महसूस होता है?",
    "यह कब शुरू हुआ?",
    "आप इस एहसास को कैसे बताएंगे — तेज़, हल्का, जलन जैसा, या कुछ और?",
    "1 से 10 के पैमाने पर, यह कितना गंभीर है?",
    "क्या आपको पहले कोई बड़ी बीमारी या सर्जरी हुई है?",
    "क्या आप फ़िलहाल कोई दवा ले रहे हैं?",
    "क्या आपको किसी दवा से एलर्जी है?",
    "क्या आपके परिवार में किसी को ऐसी या इससे जुड़ी कोई समस्या है?",
]

_AYUSH_SCRIPT = [
    ("ayush.agni", "How would you describe your appetite and digestion generally?"),
    ("ayush.nidana", "What do you think may have caused this?"),
]

_AYUSH_QUESTIONS_HI = [
    "आम तौर पर आपकी भूख और पाचन कैसा रहता है?",
    "आपको क्या लगता है इसका कारण क्या हो सकता है?",
]

_ACKNOWLEDGEMENT = {"en": "Got it, thank you.", "hi": "ठीक है, धन्यवाद।"}
_COMPLETE_MESSAGE = {
    "en": "Thanks — that's everything we need before your consultation.",
    "hi": "धन्यवाद — परामर्श से पहले हमें बस यही जानकारी चाहिए थी।",
}


def _script_for(ayush_mode: bool, language: str) -> list[tuple[str, str]]:
    fields = [f for f, _ in _CORE_SCRIPT] + ([f for f, _ in _AYUSH_SCRIPT] if ayush_mode else [])
    if language == "hi":
        questions = list(_CORE_QUESTIONS_HI) + (list(_AYUSH_QUESTIONS_HI) if ayush_mode else [])
    else:
        questions = [q for _, q in _CORE_SCRIPT] + ([q for _, q in _AYUSH_SCRIPT] if ayush_mode else [])
    return list(zip(fields, questions))


def run_turn(
    transcript: list[dict],
    ayush_mode: bool = False,
    language: str = "en",
    turn_count: int = 1,
    retry_hint: str | None = None,  # noqa: ARG001 - unused; the script never produces an empty turn, but the
    # signature has to match the other two backends so turn_runner.py can call any of them uniformly.
    current_history: str | None = None,  # noqa: ARG001 - unused
) -> TurnResult:
    script = _script_for(ayush_mode, language)

    patient_answers = [t["text"] for t in transcript if t["role"] == "patient"]

    history = ClinicalHistory(hpi=HPI(), ayush=AyushAssessment() if ayush_mode else None)
    for (field_path, _question), answer in zip(script, patient_answers):
        _set_field(history, field_path, answer)

    answered = len(patient_answers)
    if answered < len(script):
        field_path, question = script[answered]
        return TurnResult(
            history=history,
            acknowledgement=_ACKNOWLEDGEMENT.get(language, _ACKNOWLEDGEMENT["en"]) if answered > 0 else None,
            next_question=question,
            complete=False,
            red_flag=False,
        )

    return TurnResult(
        history=history,
        acknowledgement=_COMPLETE_MESSAGE.get(language, _COMPLETE_MESSAGE["en"]),
        # next_question can never be "" now (schema.py enforces min_length=1,
        # 2026-09-12 — see that field's docstring for why: an empty string
        # used to be a schema-legal way to say "done," which meant nothing
        # actually forced a real LLM backend to fill this in on a normal,
        # non-complete turn either). It's ignored downstream whenever
        # complete=True (main.py returns "question": None in that case), so
        # this value is never shown to the patient — it only exists to
        # satisfy the same non-empty contract every backend must follow.
        next_question=_COMPLETE_MESSAGE.get(language, _COMPLETE_MESSAGE["en"]),
        complete=True,
        red_flag=False,
    )


def _set_field(history: ClinicalHistory, field_path: str, value: str) -> None:
    list_fields = {
        "past_medical_surgical_history",
        "current_medications",
        "drug_allergy_history",
        "family_history",
    }
    if "." in field_path:
        obj_name, attr = field_path.split(".", 1)
        obj = getattr(history, obj_name)
        setattr(obj, attr, value)
    elif field_path in list_fields:
        getattr(history, field_path).append(value)
    else:
        setattr(history, field_path, value)
