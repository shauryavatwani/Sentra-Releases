"""Single source of truth for every path Sentra reads or writes.

Sentra runs in two very different shapes and they disagree about where files
belong:

* **Source checkout** (development, and the Mac): everything lives under the
  project folder. Reading and writing the same tree is fine.
* **Installed Windows build**: the program lives in ``C:\\Program Files\\Sentra``,
  which is read-only for a normal user. Anything the app *writes* — the
  detections database, registered faces, logs — must go somewhere else or the
  app will appear to work and then silently fail to save.

So paths are split into two roots:

``RESOURCE_ROOT``
    Read-only things shipped with the app: the dashboard HTML, the pose model,
    the InsightFace model files.

``DATA_ROOT``
    Everything the app writes: ``Database/``, ``Faces/``, ``.run_logs/``.

On a source checkout both roots are the project folder, so behaviour is exactly
what it has always been on the Mac. Only a frozen build splits them.

Set ``SENTRA_DATA_DIR`` to override the writable root (useful for putting the
client's data on a different drive).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    """True when running from a PyInstaller build rather than .py source."""
    return getattr(sys, "frozen", False)


def _resource_root() -> Path:
    if _is_frozen():
        # One-file builds extract to a temp dir exposed as _MEIPASS; one-folder
        # builds keep resources next to the exe. Support both so the packaging
        # mode can change without touching every module that reads a resource.
        bundle = getattr(sys, "_MEIPASS", None)
        if bundle:
            return Path(bundle)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    override = os.environ.get("SENTRA_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    if not _is_frozen():
        # Source checkout: keep writing into the project folder, unchanged.
        return Path(__file__).resolve().parent.parent

    if sys.platform == "win32":
        # ProgramData rather than LocalAppData: this is a shared security tool,
        # so the detection history must not be trapped inside whichever Windows
        # account happened to run the installer.
        base = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return Path(base) / "Sentra"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Sentra"
    return Path.home() / ".local" / "share" / "sentra"


RESOURCE_ROOT = _resource_root()
DATA_ROOT = _data_root()

# --- Writable ---------------------------------------------------------------

DATABASE_DIR = DATA_ROOT / "Database"
FACES_DIR = DATA_ROOT / "Faces"
# Gate photos of temporary visitors. Separate from FACES_DIR because those are
# curated enrolment photos of permanent people; these are one-shot captures
# tied to a single visit (see Formal_Code/visitor_store.py).
VISITORS_DIR = DATA_ROOT / "Visitors"
RUN_LOGS_DIR = DATA_ROOT / ".run_logs"

DETECTIONS_DB = DATABASE_DIR / "detections.db"
FACE_EMBEDDINGS_FILE = DATABASE_DIR / "face_embeddings.pkl"
CAMERA_CONFIG_FILE = DATABASE_DIR / "camera_config.json"

# Written by the engine at startup, read by the backend's restart endpoint.
# A recorded pid is what lets restart work without `pkill`, which is a
# POSIX-only tool that does not exist on Windows.
ENGINE_PID_FILE = RUN_LOGS_DIR / "engine.pid"
ENGINE_LOG_FILE = RUN_LOGS_DIR / "face_recognition.log"


def ensure_data_dirs() -> None:
    """Create the writable tree. Safe to call repeatedly."""
    for directory in (DATABASE_DIR, FACES_DIR, VISITORS_DIR, RUN_LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --- Read-only resources ----------------------------------------------------


def _first_existing(*candidates: Path) -> Path:
    """Return the first candidate that exists, else the first one.

    Falling back to the first candidate rather than raising keeps the caller's
    own error message ("model not found at X") as the thing the user sees,
    instead of an import-time crash from this module.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def pose_model_path() -> Path:
    """YOLOv8n-pose weights used by anomaly_detection.py."""
    return _first_existing(
        RESOURCE_ROOT / "Formal_Code" / "yolov8n-pose.pt",
        RESOURCE_ROOT / "models" / "yolov8n-pose.pt",
        RESOURCE_ROOT / "yolov8n-pose.pt",
    )


def insightface_root() -> Path:
    """Directory InsightFace treats as its home (it looks for ``models/`` inside).

    Passed as ``FaceAnalysis(root=...)``. A packaged build ships the ~613MB
    buffalo_l model set so the client PC needs no internet on first run; a
    source checkout keeps using the normal ``~/.insightface`` cache.
    """
    bundled = RESOURCE_ROOT / "insightface"
    if (bundled / "models").is_dir():
        return bundled
    return Path.home() / ".insightface"


def onvif_wsdl_dir() -> Path | None:
    """Directory holding the ONVIF WSDL files, or None to accept the default.

    ``onvif-zeep`` defaults this to ``<site-packages>/wsdl`` — a top-level
    directory *beside* the package rather than inside it. PyInstaller collects
    package data, not that, so a frozen build has to be told where the copy
    the spec bundled actually landed. Getting this wrong does not fail at
    import; it fails at the moment someone tries to discover a camera, with a
    file-not-found from deep inside zeep.

    Returns None when running from source, where the library's own default is
    already correct.
    """
    if not _is_frozen():
        return None
    bundled = RESOURCE_ROOT / "wsdl"
    return bundled if bundled.is_dir() else None


def app_dir() -> Path:
    """Folder holding the backend's static files (dashboard.html, login.html)."""
    return _first_existing(
        RESOURCE_ROOT / "backend_v2",
        RESOURCE_ROOT / "app",
        RESOURCE_ROOT,
    )
