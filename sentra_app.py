"""Sentra entry point — one program, two roles.

A packaged build is a single ``Sentra.exe``. Which role it plays depends on the
argument it is given:

``Sentra.exe``
    Normal launch. Starts the web server, starts the camera engine as a child
    process, and opens the dashboard in the default browser. This is what the
    Start Menu and desktop shortcuts point at.

``Sentra.exe --engine``
    Runs the camera engine only. Not something a user launches by hand — it is
    how the launcher above and the dashboard's "Restart engine" button spawn
    the engine.

Keeping both in one executable means PyInstaller builds once and there is only
one thing for the installer to ship and sign.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _fix_streams_for_windowed_build() -> None:
    """Windows + PyInstaller windowed (console=False) sets stdout/stderr to
    None, not a closed stream — there is no console to attach them to.

    That is not the same as "nothing wants to print". Uvicorn's default
    logging config builds a StreamHandler and calls ``stream.isatty()`` on it
    while deciding whether to colourise output; ``None`` has no such method,
    the call raises, and the crash happens before the dashboard has served a
    single request — reads as "the installer is broken" when it is really
    just a formatter deciding on colour codes.

    A real io stream with a no-op write is the fix: everything that expects
    to print to stdout/stderr keeps working, the bytes just go nowhere, which
    is correct for a windowed app with nothing to show them on. Must run
    before uvicorn (or anything else that touches these streams at import
    time) is imported.
    """
    if sys.stdout is None:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")


_fix_streams_for_windowed_build()

APP_HOST = "0.0.0.0"
APP_PORT = 8000
DASHBOARD_URL = f"http://localhost:{APP_PORT}"


def _bootstrap_import_path() -> None:
    """Make the Formal_Code and backend modules importable in both shapes.

    From source they are subdirectories; in a frozen build PyInstaller has
    already flattened them into the bundle, so the inserts are harmless no-ops.
    """
    root = Path(__file__).resolve().parent
    for sub in ("Formal_Code", "backend_v2"):
        candidate = root / sub
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))


_bootstrap_import_path()

import sentra_paths  # noqa: E402  (import path must be set up first)


# --------------------------------------------------------------------------
# Engine role
# --------------------------------------------------------------------------


def run_engine() -> int:
    import face_recognition

    return face_recognition.main()


# --------------------------------------------------------------------------
# Launcher role
# --------------------------------------------------------------------------


def spawn_engine() -> None:
    """Start the camera engine as a detached child process.

    Deliberately best-effort: if the engine cannot start, the dashboard must
    still come up so the user can see *why* (the Live Monitor tab reports the
    camera state) rather than getting nothing at all.
    """
    sentra_paths.ensure_data_dirs()

    if getattr(sys, "frozen", False):
        argv = [sys.executable, "--engine"]
        cwd = None
    else:
        argv = [sys.executable, str(Path(__file__).resolve()), "--engine"]
        cwd = None

    kwargs: dict = {}
    if sys.platform == "win32":
        # No console window popping up behind the dashboard.
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    try:
        with open(sentra_paths.ENGINE_LOG_FILE, "a") as logf:
            subprocess.Popen(
                argv, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT, **kwargs
            )
    except OSError as exc:
        print(f"Warning: could not start the camera engine: {exc}")


def _open_browser_when_ready() -> None:
    """Open the dashboard once the server answers, not before.

    Opening immediately races the server's startup and shows the browser's own
    connection-refused page, which reads as "the app is broken".
    """
    import urllib.error
    import urllib.request

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{DASHBOARD_URL}/login", timeout=1)
            break
        except urllib.error.HTTPError:
            break  # answering at all is enough; a 4xx still means it's up
        except OSError:
            time.sleep(0.4)
    webbrowser.open(DASHBOARD_URL)


def run_app(open_browser: bool = True) -> int:
    import uvicorn

    sentra_paths.ensure_data_dirs()
    spawn_engine()

    if open_browser:
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    import main as backend_main

    uvicorn.run(backend_main.app, host=APP_HOST, port=APP_PORT, log_level="info")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="Sentra", add_help=True)
    parser.add_argument(
        "--engine",
        action="store_true",
        help="run the camera engine only (used internally by the launcher)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the server without opening a browser window",
    )
    args = parser.parse_args()

    if args.engine:
        return run_engine()
    return run_app(open_browser=not args.no_browser)


if __name__ == "__main__":
    # Required before any child process work in a frozen build, otherwise the
    # child re-runs the launcher and forks endlessly.
    multiprocessing.freeze_support()
    raise SystemExit(main())
