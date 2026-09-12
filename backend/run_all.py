#!/usr/bin/env python3
"""
run_all.py — starts all four MediKiosk services with a single command.

This is purely a convenience wrapper for local dev / demo day. It does NOT
change how the services work: each one still runs as its own independent
FastAPI process, in its own venv, on its own port — exactly like running
them in four separate terminals. This script just does that starting (and,
on Ctrl+C, stopping) for you, and merges their logs into one place with a
[name] prefix so you can tell who said what.

If you'd rather see each service in its own window (e.g. to debug one in
isolation without the others' logs interleaved), the original four-terminal
instructions in each module's README still work unchanged — this script is
an addition, not a replacement.

Usage:
    python run_all.py

Requires each module's .venv to already exist with its requirements
installed (see the "Setup" section of converse-module/, ocr-module/,
voice-module/, and kiosk-backend/'s own READMEs) — this script starts
services, it doesn't install anything.

Stop everything with one Ctrl+C. It asks each service to shut down
cleanly, then force-kills anything still alive after a short grace period.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Order matters a little for readability of the startup log, not for
# correctness — FastAPI services don't need each other to be up yet to
# start; they just fail individual requests until their peers are ready.
SERVICES = [
    ("converse", ROOT / "converse-module", 8000),
    ("ocr", ROOT / "ocr-module", 8001),
    ("voice", ROOT / "voice-module", 8003),
    ("kiosk", ROOT / "kiosk-backend", 8002),
]

NAME_WIDTH = max(len(name) for name, _, _ in SERVICES)


def venv_python(module_dir: Path, name: str) -> str:
    """Prefer the ONE shared venv at the repo root (see requirements.txt,
    added 2026-09-11) — that's the setup this script's own README section
    now documents. Falls back to that module's own per-module .venv if the
    shared one doesn't exist, so venvs created before 2026-09-11 (the old
    "one venv per module" setup) keep working unchanged without anyone
    having to redo them. Only after both of those fails does it fall back
    to whatever python is running this script, with a loud warning — that
    last resort almost certainly doesn't have the module's dependencies
    installed, so the service will fail to start, but printing a plain
    Python traceback with a clear reason beats a silent, confusing hang.
    Windows and POSIX venv layouts are both checked at every step, since
    this is meant to run on the Windows dev machine this project actually
    gets demoed on, but shouldn't break elsewhere."""
    shared_win = ROOT / ".venv" / "Scripts" / "python.exe"
    shared_posix = ROOT / ".venv" / "bin" / "python"
    if shared_win.exists():
        return str(shared_win)
    if shared_posix.exists():
        return str(shared_posix)

    own_win = module_dir / ".venv" / "Scripts" / "python.exe"
    own_posix = module_dir / ".venv" / "bin" / "python"
    if own_win.exists():
        return str(own_win)
    if own_posix.exists():
        return str(own_posix)

    print(
        f"  ! [{name}] no shared .venv at the repo root and no per-module "
        f"venv in {module_dir} — falling back to {sys.executable}. This "
        f"will probably fail unless that python already has {name}'s "
        f"requirements installed. See the top-level README's Setup section "
        f"to create the shared venv."
    )
    return sys.executable


def stream_output(proc: subprocess.Popen, name: str) -> None:
    prefix = f"[{name:<{NAME_WIDTH}}] "
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        print(prefix + line.rstrip())


def main() -> int:
    procs: list[tuple[str, subprocess.Popen]] = []

    print("Starting all four MediKiosk services — one Ctrl+C stops all of them.\n")

    for name, module_dir, port in SERVICES:
        if not module_dir.exists():
            print(f"  ! [{name}] skipping — {module_dir} not found")
            continue
        python = venv_python(module_dir, name)
        cmd = [python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(port)]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(module_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            print(f"  ! [{name}] failed to launch: {exc}")
            continue
        procs.append((name, proc))
        threading.Thread(target=stream_output, args=(proc, name), daemon=True).start()
        time.sleep(0.3)  # stagger so the four startup banners aren't one jumbled burst

    if not procs:
        print("Nothing started — see the errors above.")
        return 1

    print(
        f"\n{len(procs)} service(s) launching. Once every line below says "
        f"'Application startup complete':\n"
        f"  patient check-in: http://localhost:8002\n"
        f"  staff/doctor view: http://localhost:8002/doctor.html (no login — see that page's own banner)\n"
    )

    try:
        while procs:
            time.sleep(1)
            still_running = []
            for name, proc in procs:
                code = proc.poll()
                if code is None:
                    still_running.append((name, proc))
                else:
                    print(f"\n  ! [{name}] exited (code {code}) — see its log above for why.")
            procs = still_running
        print("\nAll services have stopped on their own.")
        return 1
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for name, proc in procs:
            proc.terminate()
        deadline = time.time() + 5
        for name, proc in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"  ! [{name}] didn't stop in time — killing it")
                proc.kill()
        print("Done.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
