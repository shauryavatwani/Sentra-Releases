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
        with open(sentra_paths.ENGINE_LOG_FILE, "a", encoding="utf-8", errors="replace") as logf:
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


def run_selftest() -> int:
    """Prove the packaged bundle can actually do its job. Run by the build.

    Every check here corresponds to something that fails *softly* at runtime:
    the app starts, the dashboard loads, and one capability is quietly missing
    with nothing but a line in a log to say so. Fight detection dropping out
    because a torch submodule was excluded from the bundle is precisely that,
    and it shipped once.

    Turning those into a non-zero exit code means the build fails instead of
    the customer's install.
    """
    # A windowed build has no stdout, so without this the whole report would go
    # to the discarding stub and the build log would show an exit code with no
    # explanation of which capability was missing.
    _redirect_streams_to_log("selftest.log")

    failures: list[str] = []

    def check(label: str, fn) -> None:
        try:
            detail = fn()
            print(f"  [ OK ] {label}{f' — {detail}' if detail else ''}")
        except Exception as exc:  # noqa: BLE001 — reporting, not handling
            print(f"  [FAIL] {label} — {type(exc).__name__}: {exc}")
            failures.append(label)

    print(f"Sentra self-test (frozen={getattr(sys, 'frozen', False)})")

    def _pose_model():
        # Actually load it. The engine wraps this in a try/except so a failure
        # degrades to "fight detection off", which is right at runtime and
        # exactly why it has to be checked deliberately here.
        import anomaly_detection

        detector = anomaly_detection.FightDetector(label="selftest")
        model = detector.load_pose_model()
        if model is None:
            raise RuntimeError("load_pose_model() returned nothing")
        return "YOLOv8 pose + ByteTrack ready"

    def _insightface():
        # Three separate things, because each fails independently and only the
        # first was ever checked before 1.0.3.
        #
        # The model files existing proves nothing: 1.0.2 shipped a Windows
        # build whose models were all present and whose FaceAnalysis.prepare()
        # succeeded, but which could not recognise a single face — every
        # attempt died inside model.get() with "The `scipy` install you are
        # using seems to be broken (extension modules cannot be imported)".
        #
        # Nor is calling model.get() on a blank frame enough. FaceAnalysis.get()
        # returns early on `bboxes.shape[0] == 0`, so with no face in the image
        # the alignment and recognition models never run at all — which is
        # exactly the code that pulls in scipy. A blank-frame check passes on a
        # completely broken build. (1.0.3 shipped with that mistake in it.)
        #
        # face_align.norm_crop is what arcface_onnx.py calls for every detected
        # face, and it reaches skimage.transform -> scipy._lib._ccallback. Five
        # synthetic landmarks exercise that whole path with no real face needed,
        # so a build missing scipy's compiled extensions fails here in CI
        # instead of on a client's machine.
        import numpy as np
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align

        import sentra_paths as sp

        root = sp.insightface_root()
        if not (root / "models" / "buffalo_l").is_dir():
            raise RuntimeError(f"model set missing under {root}")

        model = FaceAnalysis(root=str(root), providers=["CPUExecutionProvider"])
        model.prepare(ctx_id=0, det_size=(320, 320))

        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        landmarks = np.array(
            [[80, 100], [150, 100], [115, 140], [90, 180], [145, 180]],
            dtype=np.float32,
        )
        aligned = face_align.norm_crop(canvas, landmark=landmarks, image_size=112)
        if aligned.shape != (112, 112, 3):
            raise RuntimeError(f"face alignment returned {aligned.shape}, expected (112,112,3)")

        # Belt and braces: name the module whose absence caused the 1.0.2/1.0.3
        # Windows failure, so a regression reports itself by name rather than as
        # a shape mismatch three layers up.
        import scipy._lib._ccallback  # noqa: F401

        # A second, separate bug that 1.0.3 also shipped on BOTH platforms:
        # insightface.data.get_object() reads meanshape_68.pkl from a
        # TOP-LEVEL `objects/` directory beside the bundle root when frozen —
        # not from inside the insightface package — so collect_data_files()
        # never saw it. get_object() swallows a missing file and returns None
        # rather than raising, so the failure doesn't surface until three
        # calls later, inside estimate_affine_matrix_3d23d(None, pred), as
        # "'NoneType' object has no attribute 'shape'" — for every single
        # detected face on every frozen build. norm_crop() above cannot catch
        # this: it only exercises the recognition-alignment path, never the
        # landmark_3d_68 task where this file is actually used. Checked
        # directly, by name, rather than only proven indirectly through the
        # face pipeline — a call this cheap should never be inferred.
        from insightface.data.pickle_object import get_object

        if get_object("meanshape_68.pkl") is None:
            raise RuntimeError(
                "insightface's meanshape_68.pkl reference data is missing — "
                "every detected face would fail recognition at runtime"
            )

        return f"{root} (models + face alignment + scipy + reference data OK)"

    def _websockets():
        # uvicorn without the [standard] extra serves HTTP fine and 404s every
        # WebSocket route, which kills Live Monitor and the alert banner while
        # everything else looks healthy.
        import websockets  # noqa: F401

        return "uvicorn websocket support present"

    def _onvif():
        import sentra_paths as sp
        from onvif import ONVIFCamera  # noqa: F401

        wsdl = sp.onvif_wsdl_dir()
        if getattr(sys, "frozen", False) and wsdl is None:
            raise RuntimeError("bundled WSDL directory not found")
        return str(wsdl) if wsdl else "using package default"

    def _backend():
        import main as backend_main

        routes = [r.path for r in backend_main.app.routes]
        for required in ("/api/stats", "/ws/live", "/api/data-pack/import"):
            if required not in routes:
                raise RuntimeError(f"route missing: {required}")
        return f"{len(routes)} routes"

    def _data_pack():
        import data_pack  # noqa: F401

        return "import/export available"

    def _ssl():
        import ssl

        ssl.create_default_context()
        return "update check can reach https"

    check("InsightFace model set", _insightface)
    check("Fight detection (pose model)", _pose_model)
    check("WebSocket support", _websockets)
    check("ONVIF discovery", _onvif)
    check("Backend routes", _backend)
    check("Data pack", _data_pack)
    check("TLS", _ssl)

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s) — {', '.join(failures)}")
        return 1
    print("SELF-TEST PASSED")
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
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="verify this build's capabilities and exit (used by the build scripts)",
    )
    args = parser.parse_args()

    if args.selftest:
        return run_selftest()
    if args.engine:
        return run_engine()
    return run_app(open_browser=not args.no_browser)


if __name__ == "__main__":
    # Required before any child process work in a frozen build, otherwise the
    # child re-runs the launcher and forks endlessly.
    multiprocessing.freeze_support()
    raise SystemExit(main())
