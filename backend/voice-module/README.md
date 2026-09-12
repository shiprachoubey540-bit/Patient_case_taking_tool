# MediKiosk — Voice / Speech-to-Text

The "voice" role the Converse module's own README anticipated but
deliberately didn't build: "voice lead wires that in front of
`POST /sessions/{id}/message` — this module never touches audio." This is
that module. It takes a recorded audio clip and returns plain text; it has
no opinion about clinical history, that's still entirely the Converse
module's job.

**What this module does NOT do** — other owners' modules per the build plan:
- No conversational logic — hand its output to the Converse module's
  `/sessions/{id}/message`, same as if the patient had typed it.
- No text-to-speech (reading questions aloud). Not built — a natural
  follow-on if you want a fully hands-free kiosk later.
- No real-time/streaming captioning. This is record-a-clip, upload,
  get-text-back — not a live transcript while the patient is still talking.
- No real kiosk UI. `static/index.html` is a bare test harness (record,
  see the transcript) — the actual mic button lives in `kiosk-backend`.

## Setup

**As of 2026-09-11, the whole project shares one venv** — see the
[top-level README](../README.md)'s Setup section and just run that once;
it covers this module too (and that's where the `faster-whisper` install
time warning below now lives too). That's the recommended path.

Only if you want to run *this module alone*, isolated from the other
three (its own venv, nothing shared) — still works exactly as before:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This install is heavier than the other modules' — `faster-whisper` pulls in
`ctranslate2` and a few other sizeable packages. Give it a few minutes on
first install.

No API key needed — this runs entirely locally. Copy `.env.example` to
`.env` if you want to change the model size or force CPU/GPU (see below);
the defaults work with zero configuration.

**The first real transcription will download the Whisper model** (~500MB
for the default "small" size) from Hugging Face — that needs network once.
Every transcription after that is fully offline, and no audio ever leaves
the machine. This matters more here than almost anywhere else in this
project: it's the patient's actual recorded voice, not just typed text —
treat test recordings the same way the OCR module's README asks you to
treat test documents: don't record and upload anyone's real medical
complaint during development, make something up.

Runs in **mock mode** with zero setup (no model download needed) — same
pattern as the other modules: a fixed canned transcript regardless of what
audio you send, so the rest of the pipeline (upload, validation, the kiosk's
mic button wiring) can be built and tested without the ~500MB download or
waiting on transcription.

## Run it

```bash
uvicorn app.main:app --reload --port 8003
```

Open http://localhost:8003 for the test page: click record, say something,
click stop, watch the transcript appear. Needs mic permission in the
browser (works on `localhost` without HTTPS).

Or hit the API directly:

```bash
curl localhost:8003/backends

curl -X POST localhost:8003/transcribe \
  -F "file=@clip.webm" \
  -F "backend=auto"
```

## Run the tests

```bash
pytest tests/ -v
```

Entirely against the mock backend, with arbitrary bytes standing in for
real audio — no network, no model download, no microphone needed. The
auto-prefers-whisper / whisper-rejected-when-unavailable resolution logic
is tested by monkeypatching `whisper_available()` and
`whisper_backend.transcribe()`, not by actually loading a model.

**Important caveat on what these tests do NOT prove**: the environment this
module was built in had no network access to huggingface.co, so an actual
model download and a real transcription of real speech could never be run
end-to-end during development — only the `faster-whisper` library's API
surface was checked (import succeeds, the function signatures this code
calls match what the installed package actually exposes). Whether
transcription quality is good enough to use, whether the browser's
recorded `audio/webm` format decodes cleanly, and whether CPU latency is
acceptable on your hardware all still need a real test on a machine with
network access and a working microphone — same category of "needs your
live verification" as the Ollama backend did.

## Design decisions worth knowing about

- **CPU by default, not GPU.** `WHISPER_DEVICE=cpu` — deliberately, so this
  doesn't compete with Ollama (or whatever LLM you're running) for VRAM on
  the same machine. Whisper models are much smaller than LLMs; a short
  patient utterance should transcribe in low single-digit seconds on CPU
  with the default "small" model on ordinary laptop hardware. Set
  `WHISPER_DEVICE=cuda` (and `WHISPER_COMPUTE_TYPE=float16`) if you'd
  rather trade VRAM headroom for speed — not recommended while an LLM is
  also loaded, unless you've checked you have room for both.
- **Multilingual by default.** Whisper isn't English-only — `language` is
  left unset (auto-detect) unless the caller passes a hint. Worth actually
  testing with a non-English clip if you plan to demo this as a real
  feature for patients who don't speak English, rather than assuming it
  works because the model card says it supports the language.
- **VAD (voice-activity detection) filtering is on.** Skips silence/
  background noise instead of transcribing it — reduces the chance of the
  model hallucinating text from a quiet kiosk-lobby recording.
- **No audio format validation on upload**, unlike the OCR module's
  Pillow-based image check. There's no equally simple universal audio
  validator available, so a malformed or empty file just fails at
  transcription time with a clear 502 rather than being caught earlier at
  upload time. Known gap, not a design principle.
- **Same pluggable-backend shape as the other modules.**
  `whisper_backend.py` and `mock_transcribe.py` both expose
  `transcribe(audio_bytes, language) -> dict`, so `main.py` doesn't care
  which one is active.

## Known gaps — be upfront about these if asked

- Transcription accuracy on real recordings is unverified (see the
  caveat above) — don't claim this works reliably until you've tested it
  yourself with real speech, background noise, and an actual accent.
- No audio format validation, no size-based rejection beyond a raw byte
  cap (15MB).
- No retry logic if a transcription fails — the kiosk frontend should treat
  a failure here as "let the patient type instead," not retry silently.
- No persistence — nothing is stored; a transcript is handed back once and
  the audio itself is discarded (never written to disk here). Worth
  confirming that's still true if you extend this module.

## Model name / library

`faster-whisper` (CTranslate2-based). `WHISPER_MODEL` defaults to `small`.
Check https://github.com/SYSTRAN/faster-whisper for current model options
and any interface changes if it's been a while since this was written.
