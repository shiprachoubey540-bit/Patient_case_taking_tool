# Module C (stub) — Case Summary Generator

A working stand-in for Module C so you can demo the full A→B→C flow
even before your teammate's real Module C is ready. It calls your
Module B `/digitize` endpoint and merges the result with a placeholder
conversation-history payload (standing in for Module A).

## Setup (same pattern as Module B)
Put this folder as a sibling to your `sih` module_b folder, e.g.:
```
C:\Users\Shipra\sih\        <- module B (already running)
C:\Users\Shipra\sih_c\      <- this folder
```

```powershell
cd C:\Users\Shipra\sih_c
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
Module B must already be running on port 8001. In a **second** terminal:
```powershell
uvicorn app.main:app --reload --port 8002
```

Open http://localhost:8002/docs

## Testing the merge
1. Expand `POST /case-summary`, click "Try it out".
2. Upload the same test document you used for Module B.
3. In the `conversation_history` field, you can leave the default
   placeholder, or paste something like:
   ```json
   {"chief_complaint": "fever and cough", "symptoms": ["fever", "cough", "fatigue"], "duration": "3 days", "reported_medications": ["paracetamol"]}
   ```
4. Execute — you'll get back one merged JSON combining both sources.

## When your teammate's real Module A/C are ready
- Update `normalize_conversation_history()` in `app/merge_logic.py` to
  match Module A's actual output schema (only that one function needs
  to change).
- If a teammate builds the real Module C, this stub becomes optional —
  keep it as a fallback / testing tool for your own demo runs.
