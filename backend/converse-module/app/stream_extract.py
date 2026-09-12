"""
Incremental extraction of specific string field values out of a JSON object
as it streams in, character-by-character, from an LLM's structured-output
generation (Ollama's `/api/chat` with `stream: true`).

Why this exists (2026-09-12, the "live tokens" feature): TurnResult's schema
puts a large nested `history` object FIRST, then `acknowledgement`, then
`next_question` — the two fields a patient actually needs to see (see
schema.py). Naively piping the raw model output straight to the UI would
show the patient raw JSON scaffolding (field names, braces, the entire
history object being filled in field by field) before anything readable
appears — arguably worse than a plain "thinking..." spinner, and it would
also leak the internal schema shape to the UI layer. This module instead
watches the raw stream for exactly the target fields' string VALUES and
yields only their plain-text (already JSON-unescaped) characters as they
arrive, staying silent while the model is elsewhere in the JSON.

Honest limitation worth knowing before this gets credited with more than it
delivers: because `history` is serialized first, the model still has to
generate that entire (often large) object before emitting a single visible
character — so this does NOT shorten the initial wait before anything shows
up. What it does do is turn the LAST stretch of a turn (writing
acknowledgement + next_question, once the model gets there) from "appears
all at once" into "appears progressively," which is a real but partial win.
Reordering TurnResult's fields so acknowledgement/next_question come first
would remove that initial wait too, but at a real cost: it would mean the
model commits to its answer before "showing its work" via the structured
history extraction, which is a plausible quality regression (this is the
same reasoning-before-answering tradeoff chain-of-thought prompting relies
on) — not done here without deliberately testing that tradeoff first.

This is NOT a general JSON streaming parser — it doesn't track nesting,
validate structure, or understand anything about the schema beyond "find
this key, then read its string value (or notice it's null and move on)".
That's deliberately as far as this needs to go for TurnResult's flat
top-level Optional[str]/str fields.

Escaping: JSON string escapes (\\n, \\", \\\\, \\uXXXX, etc.) are decoded
before being handed back, and a chunk boundary landing mid-escape-sequence
(or mid-key, or right at "null" vs a string's opening quote) is handled by
holding back rather than guessing — feed() only ever emits text it's fully
sure of.
"""
from __future__ import annotations

import re
from enum import Enum, auto
from typing import List, Tuple

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class _State(Enum):
    SEEKING_KEY = auto()      # looking for "<current target>":
    AWAITING_VALUE = auto()   # found the key/colon, deciding string vs null
    IN_STRING = auto()        # inside the target's string value, streaming chars out
    DONE = auto()             # no targets left


class IncrementalFieldExtractor:
    """Give it the ordered list of top-level string field names you expect
    to see in the JSON object (in schema-declared order), feed it raw text
    chunks as they arrive, and it tells you which target field's text grew
    and by how much. Fields the model sets to JSON null (Optional[str] ->
    None) are skipped silently, same as if they'd never matched."""

    def __init__(self, field_names: List[str]):
        self._targets = list(field_names)
        self._target_idx = 0
        self._buffer = ""
        self._scan_from = 0  # safe resume point for the SEEKING_KEY regex search
        self._key_match_end = None  # index right after "<key>": once matched, awaiting the value
        self._cursor = 0  # index into _buffer where IN_STRING scanning resumes
        self._state = _State.SEEKING_KEY if self._targets else _State.DONE

    def _current_target(self):
        if self._target_idx >= len(self._targets):
            return None
        return self._targets[self._target_idx]

    def _advance_target(self, resume_from: int) -> None:
        self._target_idx += 1
        self._scan_from = resume_from
        self._key_match_end = None
        self._state = _State.SEEKING_KEY if self._current_target() else _State.DONE

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        """Feed a new chunk of raw model output. Returns a list of
        (field_name, text_delta) pairs for this call — usually zero or one,
        but can be more than one if this chunk both finished one target
        field and produced text for the next."""
        self._buffer += chunk
        out: List[Tuple[str, str]] = []

        while True:
            if self._state == _State.DONE:
                return out

            target = self._current_target()

            if self._state == _State.SEEKING_KEY:
                pattern = re.compile(r'"' + re.escape(target) + r'"\s*:\s*')
                match = pattern.search(self._buffer, self._scan_from)
                if match is None:
                    # This target's key might simply never appear (e.g. an
                    # Optional[str] field the model omitted outright rather
                    # than writing as null) — relying only on "keep waiting"
                    # would get stuck here forever and never reach a LATER
                    # target that's already sitting in the buffer. This
                    # module only serves TurnResult's fixed, small field
                    # list, and relies on the model emitting properties in
                    # schema-declared order (a safe assumption for
                    # grammar-constrained generation from one fixed schema,
                    # not something a general-purpose parser could assume).
                    # So: if any LATER target's key is already found, this
                    # one was skipped — jump ahead to it instead of waiting
                    # on a key that will never arrive.
                    skip_to = None
                    for later_idx in range(self._target_idx + 1, len(self._targets)):
                        later_pattern = re.compile(r'"' + re.escape(self._targets[later_idx]) + r'"\s*:\s*')
                        if later_pattern.search(self._buffer, self._scan_from):
                            skip_to = later_idx
                            break
                    if skip_to is not None:
                        self._target_idx = skip_to
                        self._key_match_end = None
                        continue  # retry SEEKING_KEY immediately for the new current target
                    # Keep scan_from a little behind the end in case the key
                    # itself is split across a chunk boundary.
                    self._scan_from = max(0, len(self._buffer) - (len(target) + 8))
                    return out
                self._key_match_end = match.end()
                self._state = _State.AWAITING_VALUE
                # fall through

            if self._state == _State.AWAITING_VALUE:
                pos = self._key_match_end
                # The key-matching regex's trailing \s* only accounts for
                # whitespace that had already arrived by the time it ran —
                # more whitespace can still show up here one character at a
                # time (e.g. the space in `"acknowledgement": "ok"` arriving
                # in its own feed() call, separately from the colon before
                # it). Skip past any of it before deciding what the value
                # actually starts with; remember how far we've confirmed so
                # a later call doesn't redo this scan.
                n = len(self._buffer)
                while pos < n and self._buffer[pos] in " \t\r\n":
                    pos += 1
                self._key_match_end = pos
                available = n - pos
                if available == 0:
                    return out  # don't know yet whether it's a string or null
                if self._buffer[pos] == '"':
                    self._cursor = pos + 1
                    self._state = _State.IN_STRING
                    # fall through to scan the value immediately
                elif self._buffer[pos] != "n":
                    # Not a string and not the start of "null" — something
                    # unexpected (a number, object, etc). Not handled; treat
                    # the field as absent rather than guessing at its shape.
                    self._advance_target(pos)
                    continue
                else:
                    if available < 4:
                        return out  # could still be "null" — wait for more data
                    if self._buffer[pos:pos + 4] == "null":
                        self._advance_target(pos + 4)
                        continue
                    # Starts with 'n' but isn't "null" — unexpected, same as above.
                    self._advance_target(pos)
                    continue

            # _State.IN_STRING: scan forward for an unescaped closing quote,
            # decoding escapes as we go.
            decoded: List[str] = []
            i = self._cursor
            n = len(self._buffer)
            closed = False
            while i < n:
                ch = self._buffer[i]
                if ch == "\\":
                    if i + 1 >= n:
                        break  # incomplete escape — wait for more data
                    esc = self._buffer[i + 1]
                    if esc == "u":
                        if i + 6 > n:
                            break  # incomplete \uXXXX — wait for more data
                        hex4 = self._buffer[i + 2 : i + 6]
                        try:
                            decoded.append(chr(int(hex4, 16)))
                        except ValueError:
                            pass  # malformed escape — drop rather than crash a live demo
                        i += 6
                    elif esc in _SIMPLE_ESCAPES:
                        decoded.append(_SIMPLE_ESCAPES[esc])
                        i += 2
                    else:
                        # Unknown escape — keep both characters verbatim
                        # rather than silently losing data.
                        decoded.append(ch)
                        decoded.append(esc)
                        i += 2
                elif ch == '"':
                    closed = True
                    i += 1
                    break
                else:
                    decoded.append(ch)
                    i += 1

            delta = "".join(decoded)
            if delta:
                out.append((target, delta))
            self._cursor = i

            if closed:
                self._advance_target(i)
                continue  # a later target might already be sitting in the buffered text
            return out  # ran out of buffer mid-value; wait for the next chunk
