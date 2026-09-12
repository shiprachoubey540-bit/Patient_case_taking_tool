"""
Fast, rule-based red-flag pre-check.

Design decision: don't rely solely on the LLM to catch emergency symptoms.
This keyword check runs on every patient message BEFORE the LLM call, so an
emergency is flagged even if the model call fails, times out, or simply
misjudges a turn. The LLM's own red-flag judgment (see prompts.py) is a
second, complementary layer — not a replacement for this one.

This list is a starting point, not a clinically validated one. Treat it the
same way as the AYUSH questions: get a clinician to review and extend it
before this goes anywhere near a real patient.

Multilingual note (added 2026-09-07): the check below runs against ALL known
languages' keywords regardless of which language the session was started
in — deliberately not gated by session.language. Two reasons: (1) this is a
safety net, and a false positive here just costs a moment of staff
attention, while a false negative could genuinely matter, so over-inclusion
is the right failure mode; (2) patients code-switch — someone who picked
"Hindi" at check-in may still type or say an English phrase like "chest
pain" (or vice versa), and gating the check by the UI's language setting
would silently miss that. RED_FLAG_KEYWORDS_HI is a Claude-generated
translation, not reviewed by a native or clinical Hindi speaker — same
caveat as everything else added on 2026-09-07 here and in prompts.py.
"""
from __future__ import annotations

from typing import Optional

RED_FLAG_KEYWORDS_EN = [
    # --- cardiac / respiratory ---
    "chest pain",
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "difficulty breathing",
    "shortness of breath",
    "choking",
    # --- bleeding ---
    "severe bleeding",
    "heavy bleeding",
    "coughing blood",
    "vomiting blood",
    # --- neuro / stroke ---
    "unconscious",
    "fainted",
    "seizure",
    "slurred speech",
    "face drooping",
    "one side of my face",
    "sudden weakness",
    "can't move my",
    "cant move my",
    "sudden numbness",
    "sudden confusion",
    "worst headache of my life",
    "sudden severe headache",
    # --- mental health crisis ---
    "suicidal",
    "want to die",
    "kill myself",
    # --- abdominal ---
    "severe abdominal pain",
    # --- allergic reaction / anaphylaxis (added 2026-09-07, unreviewed —
    # see module docstring: get a clinician to check these before relying
    # on them) ---
    "anaphylaxis",
    "severe allergic reaction",
    "throat is closing",
    "throat closing up",
    "swelling of my throat",
    "swollen tongue",
    "tongue is swelling",
    # --- poisoning / overdose (added 2026-09-07, unreviewed) ---
    "overdose",
    "swallowed poison",
    "took too many pills",
    "took too many tablets",
    # --- meningitis-pattern (added 2026-09-07, unreviewed — "stiff neck"
    # alone is a broad net; a false positive here just means an earlier
    # staff look, not a wrong diagnosis, but flagging the tradeoff) ---
    "stiff neck",
    "neck stiffness",
    # --- eyes (added 2026-09-07, unreviewed) ---
    "sudden vision loss",
    "lost vision in",
    "can't see out of",
    # --- pregnancy (added 2026-09-07, unreviewed) ---
    "bleeding and i'm pregnant",
    "pregnant and bleeding",
    "baby not moving",
    "baby stopped moving",
    "reduced fetal movement",
    # --- pediatric (added 2026-09-07, unreviewed — this module doesn't
    # track patient age, so these rely on the caregiver's own wording) ---
    "baby has a high fever",
    "infant fever",
    "newborn fever",
    # --- burns / trauma (added 2026-09-07, unreviewed) ---
    "severe burn",
    "testicular pain",
    "twisted testicle",
]

# Hindi equivalents of the groups above (added 2026-09-07 alongside
# multilingual interview support — see module docstring: unreviewed
# translation, not clinically validated). Devanagari has no case, so the
# .lower() call in keyword_red_flag_check() below is a harmless no-op on
# these — no separate normalization needed.
RED_FLAG_KEYWORDS_HI = [
    # --- cardiac / respiratory ---
    "सीने में दर्द",
    "छाती में दर्द",
    "सांस नहीं आ रही",
    "सांस लेने में तकलीफ",
    "दम घुट रहा है",
    # --- bleeding ---
    "बहुत खून बह रहा है",
    "ज़्यादा खून बह रहा है",
    "खून की खांसी",
    "खून की उल्टी",
    # --- neuro / stroke ---
    "बेहोश",
    "दौरा पड़ा",
    "मिर्गी का दौरा",
    "बोलने में दिक्कत",
    "चेहरा एक तरफ झुक",
    "अचानक कमज़ोरी",
    "हिल नहीं पा रहा",
    "हिल नहीं पा रही",
    "सुन्न पड़ गया",
    "अचानक भ्रम",
    "ज़िंदगी का सबसे तेज़ सिरदर्द",
    "अचानक तेज़ सिरदर्द",
    # --- mental health crisis ---
    "आत्महत्या",
    "मरना चाहता हूं",
    "मरना चाहती हूं",
    "खुद को मारना",
    # --- abdominal ---
    "पेट में तेज़ दर्द",
    # --- allergic reaction / anaphylaxis ---
    "गंभीर एलर्जी",
    "गला बंद हो रहा है",
    "जीभ में सूजन",
    # --- poisoning / overdose ---
    "ज़हर खा लिया",
    "ज़्यादा दवा खा ली",
    "ओवरडोज़",
    # --- meningitis-pattern ---
    "गर्दन में अकड़न",
    # --- eyes ---
    "अचानक दिखना बंद",
    "एक आंख से दिखना बंद",
    # --- pregnancy ---
    "गर्भवती हूं और खून बह रहा है",
    "बच्चा हिल नहीं रहा",
    # --- pediatric ---
    "बच्चे को तेज़ बुखार",
    "नवजात को बुखार",
    # --- burns / trauma ---
    "गंभीर जलन",
]

RED_FLAG_KEYWORDS = RED_FLAG_KEYWORDS_EN + RED_FLAG_KEYWORDS_HI


def keyword_red_flag_check(text: str) -> Optional[str]:
    """Return the matched phrase, or None if nothing matched. Checked
    against every language's keyword list regardless of the session's own
    language setting — see the multilingual note in the module docstring."""
    low = text.lower()
    for phrase in RED_FLAG_KEYWORDS:
        if phrase in low:
            return phrase
    return None
