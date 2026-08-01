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
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


class _NullWriter(io.TextIOBase):
    """A writable stream that discards, but answers every question asked of it.

    Deliberately not ``io.TextIOWrapper(io.BytesIO())``: that keeps every byte
    ever written, and this process is a server that logs a line per request and
    runs for weeks. It would be an invisible memory leak.

    ``isatty`` is the method whose absence caused the original crash, but any
    of ``fileno``/``flush``/``write`` can be probed by a logging handler, so
    all of them answer rather than raising.
    """

    def write(self, text):  # noqa: D102
        return len(text)

    def flush(self):  # noqa: D102
        return None

    def isatty(self):  # noqa: D102
        return False

    def writable(self):  # noqa: D102
        return True


def _fix_streams_for_windowed_build() -> None:
    """Windows + PyInstaller windowed (console=False) sets stdout/stderr to
    None, not a closed stream — there is no console to attach them to.

    That is not the same as "nothing wants to print". Uvicorn's default logging
    config builds a StreamHandler and calls ``stream.isatty()`` while deciding
    whether to colourise output; ``None`` has no such method, so the process
    dies with "Unable to configure formatter 'default'" before serving a single
    request — which reads as a broken installer rather than as a formatter
    picking colour codes.

    This is the earliest possible safety net. :func:`_redirect_streams_to_log`
    replaces it with a real file once the data directory is known, so the output
    is actually diagnosable; this only has to survive until then.

    A stream handed over by a parent process (the engine, whose stdout the
    launcher redirects into a log file) is left alone — it is already valid.
    """
    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()


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

# A windowed build has nowhere to show output, so without this every traceback
# the app produces is lost — including the ones that explain why it will not
# start. 5MB is roughly a week of ordinary request logging.
MAX_LOG_BYTES = 5 * 1024 * 1024


def _redirect_streams_to_log(log_name: str) -> None:
    """Point discarded output at a real file, now that DATA_ROOT is known.

    Only replaces streams that :func:`_fix_streams_for_windowed_build` stubbed
    out. A console build keeps its console; the engine keeps the log handle its
    parent gave it.

    Failure here is deliberately swallowed: not being able to open a log file
    (a locked file, a full disk) is not a reason to refuse to run the security
    system. The stub stays in place and the app carries on silently.
    """
    if not isinstance(sys.stdout, _NullWriter) and not isinstance(sys.stderr, _NullWriter):
        return
    try:
        sentra_paths.ensure_data_dirs()
        path = sentra_paths.RUN_LOGS_DIR / log_name
        # Truncate rather than rotate: this is a diagnostic tail, not an audit
        # trail, and an unbounded log on a machine nobody administers is its
        # own bug.
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            path.unlink()
        # line_buffering so a crash still leaves the lines that led up to it —
        # a block-buffered log loses exactly the part you need.
        handle = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return

    if isinstance(sys.stdout, _NullWriter):
        sys.stdout = handle
    if isinstance(sys.stderr, _NullWriter):
        sys.stderr = handle
    print(f"\n--- Sentra started {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


# --------------------------------------------------------------------------
# Engine role
# --------------------------------------------------------------------------


def run_engine() -> int:
    # Only bites if the engine was started without the launcher's redirect;
    # normally its parent has already handed it a log file.
    _redirect_streams_to_log("face_recognition.log")
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

    # Python block-buffers stdout when it is a file rather than a terminal, so
    # a perfectly healthy engine writes nothing to its log for minutes at a
    # time and only flushes when it crashes — which has cost real debugging
    # time on this project before. From source the fix is `python3 -u`; a
    # frozen build has no command line to put that on, so it goes in the
    # environment, which Python honours either way.
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    try:
        with open(sentra_paths.ENGINE_LOG_FILE, "a") as logf:
            subprocess.Popen(
                argv, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT, env=env, **kwargs
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


def _tell_user(message: str, title: str = "Sentra") -> None:
    """Say something to a user who has no console to read it in.

    A windowed build's ``print`` goes to a log file nobody opens. When the
    reason the app is not starting is something the user can act on, it has to
    be on screen. Falls back to printing where there is no message box.
    """
    print(message)
    if sys.platform == "win32":
        try:
            import ctypes

            MB_OK, MB_ICONINFORMATION = 0x0, 0x40
            ctypes.windll.user32.MessageBoxW(
                None, message, title, MB_OK | MB_ICONINFORMATION
            )
        except Exception:  # noqa: BLE001 — never let a dialog failure escalate
            pass


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.6)
        # Connecting rather than binding: binding to 0.0.0.0 can succeed on
        # Windows even when another process holds the same port on a specific
        # interface, which would let this check pass and uvicorn still fail.
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _sentra_already_running() -> bool:
    """True when the thing holding our port is a Sentra that is already up."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{DASHBOARD_URL}/login", timeout=2)
        return True
    except urllib.error.HTTPError:
        return True  # answering at all means a server is there
    except OSError:
        return False


def run_app(open_browser: bool = True) -> int:
    # Before uvicorn is imported: its logging config inspects these streams at
    # configuration time, and this is what makes a startup failure readable
    # afterwards instead of vanishing.
    _redirect_streams_to_log("sentra.log")

    # Double-clicking the icon twice is the single most likely way a user meets
    # this: uvicorn would abort with "address already in use" and, having no
    # console, the second copy would appear to do nothing at all. Opening the
    # dashboard that is already running is what the user actually wanted.
    if _port_in_use(APP_HOST, APP_PORT):
        if _sentra_already_running():
            print("Sentra is already running; opening the existing dashboard.")
            if open_browser:
                webbrowser.open(DASHBOARD_URL)
            return 0
        _tell_user(
            f"Sentra could not start because another program is already using "
            f"port {APP_PORT} on this PC.\n\n"
            f"Close that program and start Sentra again. If you are not sure "
            f"what it is, restarting the PC will clear it.",
            "Sentra — port in use",
        )
        return 1

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
