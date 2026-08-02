# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Sentra (Windows).

Run from the project root:

    pyinstaller windows/sentra.spec --noconfirm

Produces a one-FOLDER build in dist/Sentra/. One-folder, not one-file, on
purpose: torch and the InsightFace models make this a multi-GB app, and a
one-file build re-extracts all of it to a temp directory on every single
launch — a startup delay measured in minutes, plus double the disk use.
Inno Setup packs the folder into the actual installer afterwards.
"""

import datetime
import json
import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH).resolve().parent

# --- Version stamping -----------------------------------------------------
# The version literal lives in Formal_Code/sentra_version.py and nowhere else.
# Read it here (by regex rather than by importing, so the spec does not need
# Sentra's runtime dependencies installed) and hand it to two places:
#
#   build_info.json      -> read at runtime, shown in Settings > About
#   version_define.iss   -> #included by sentra.iss, so the installer's version
#                           can never drift from the app's
#
# Bumping one number in sentra_version.py therefore updates the About panel,
# the installer metadata, and what the update check compares against.

_version_src = (PROJECT_ROOT / "Formal_Code" / "sentra_version.py").read_text(encoding="utf-8")
_match = re.search(r'^VERSION\s*=\s*"([^"]+)"', _version_src, re.MULTILINE)
if not _match:
    raise SystemExit(
        "\nERROR: could not read VERSION from Formal_Code/sentra_version.py.\n"
        "The installer and the About panel both take their version from there.\n"
    )
APP_VERSION = _match.group(1)
BUILD_DATE = datetime.date.today().isoformat()

BUILD_INFO = PROJECT_ROOT / "windows" / "build_info.json"
BUILD_INFO.write_text(
    json.dumps(
        {
            "version": APP_VERSION,
            "build_date": BUILD_DATE,
            "channel": "release",
            "platform": sys.platform,
        },
        indent=2,
    ),
    encoding="utf-8",
)

# Inno's #define syntax. Generated rather than hand-maintained so the two
# version numbers cannot disagree.
(PROJECT_ROOT / "windows" / "version_define.iss").write_text(
    f'#define AppVersion "{APP_VERSION}"\n'
    f'#define BuildDate "{BUILD_DATE}"\n',
    encoding="utf-8",
)

print(f"[sentra.spec] Building Sentra {APP_VERSION} (build date {BUILD_DATE})")

# --- Model files ----------------------------------------------------------
# These are the two things the app cannot download at the client's site, so a
# build that silently omits them produces an installer that looks fine and then
# fails on the client's PC. Fail the build here instead.

INSIGHTFACE_MODELS = Path(
    os.environ.get("SENTRA_INSIGHTFACE_DIR", Path.home() / ".insightface")
)
BUFFALO = INSIGHTFACE_MODELS / "models" / "buffalo_l"
if not BUFFALO.is_dir():
    raise SystemExit(
        f"\nERROR: InsightFace model set not found at:\n    {BUFFALO}\n\n"
        "The installer must ship these (~613MB) so the client PC needs no\n"
        "internet on first run. Either run the app once on this machine to\n"
        "download them, or copy the .insightface folder from the dev Mac, or\n"
        "point SENTRA_INSIGHTFACE_DIR at wherever they live.\n"
    )

POSE_WEIGHTS = PROJECT_ROOT / "Formal_Code" / "yolov8n-pose.pt"
if not POSE_WEIGHTS.is_file():
    raise SystemExit(
        f"\nERROR: pose model not found at:\n    {POSE_WEIGHTS}\n\n"
        "Without it fight detection silently disables itself.\n"
    )

# The face --selftest recognises to prove this build works. Without it the
# self-test cannot run, and the self-test is the only thing standing between a
# silently-broken bundle and a customer's machine — so a missing fixture fails
# the build rather than quietly weakening the gate.
SELFTEST_FACE = PROJECT_ROOT / "tests" / "fixtures" / "selftest_face.jpg"
if not SELFTEST_FACE.is_file():
    raise SystemExit(
        f"\nERROR: self-test face image not found at:\n    {SELFTEST_FACE}\n\n"
        "--selftest needs it to prove face recognition actually works in the\n"
        "built bundle. Without it the build ships unverified.\n"
    )

# --- Bundled data ---------------------------------------------------------

datas = [
    # Dashboard + login are served straight off disk by main.py.
    (str(PROJECT_ROOT / "backend_v2" / "dashboard.html"), "backend_v2"),
    (str(PROJECT_ROOT / "backend_v2" / "login.html"), "backend_v2"),
    # Build stamp, read by sentra_version.py at runtime. Lands at the resource
    # root so RESOURCE_ROOT / "build_info.json" resolves in both build shapes.
    (str(BUILD_INFO), "."),
    # Pose weights. anomaly_detection.py resolves this via sentra_paths.
    (str(POSE_WEIGHTS), "Formal_Code"),
    # InsightFace expects <root>/models/buffalo_l/*.onnx
    (str(BUFFALO), "insightface/models/buffalo_l"),
    # 14KB synthetic face --selftest runs through the full recognition
    # pipeline. Bundled deliberately: every check that did not put a real face
    # through model.get() passed on builds that could not recognise anyone.
    (str(SELFTEST_FACE), "tests/fixtures"),
]

# ultralytics ships yaml configs it reads at runtime; without these the pose
# model fails to build its head. Same story for the insightface package data.
datas += collect_data_files("ultralytics")
datas += collect_data_files("insightface")
# zeep (the SOAP stack under onvif-zeep) reads its own XSDs at runtime.
datas += collect_data_files("zeep")
datas += collect_data_files("onvif")  # onvif/version.txt

# scipy + scikit-image, Windows-only gap (found 2026-08-02, "face detection
# works on macOS, not on Windows"): InsightFace's face-alignment step lazily
# imports scipy._lib._ccallback the first time a real face is aligned — not at
# model load, so a build that starts, serves the dashboard, and passes every
# --selftest check up to this point still fails the instant a face appears,
# with "The `scipy` install you are using seems to be broken (extension
# modules cannot be imported)". Static import analysis never sees that lazy
# import, and scipy's Windows wheels additionally load their OpenBLAS runtime
# from a *sibling* DLL folder via `_distributor_init.py`'s DLL-search-path
# trick rather than a Python import — collect_submodules() alone cannot find
# binaries reached that way. macOS wheels bundle the equivalent differently
# (no external DLL step), which is why this never showed up on the Mac build.
# collect_all() pulls submodules + data files + binaries in one call — the
# same thing the community scipy/scikit-image PyInstaller hooks do — added
# explicitly here as insurance against a hook-version gap rather than trusting
# it silently. Cheap: purely additive, cannot break an already-working build.
_scipy_datas, _scipy_binaries, _scipy_hidden = collect_all("scipy")
_skimage_datas, _skimage_binaries, _skimage_hidden = collect_all("skimage")
datas += _scipy_datas + _skimage_datas

# insightface.data.get_object() reads its pickled reference data (currently
# just meanshape_68.pkl, used by the landmark_3d_68 model every detected face
# goes through) from a TOP-LEVEL `objects/` directory beside sys._MEIPASS when
# frozen — not from inside the insightface package, so collect_data_files
# above (which preserves the package's own internal layout) does not see it.
# Exactly the same sibling-directory trap as onvif-zeep's WSDLs below.
#
# Unlike the ONVIF case this is not a soft "feature unavailable": get_object()
# swallows the missing file and returns None, so the failure surfaces three
# calls later as `'NoneType' object has no attribute 'shape'` inside
# insightface.utils.transform.estimate_affine_matrix_3d23d, for every single
# detected face, on every frozen build — the "the live feed works but nobody
# is ever recognised" bug. Found 2026-08-02 by tracing that exact traceback;
# it was invisible before because the Windows scipy bug (above) always failed
# first, and the packaged macOS build had never actually reached a live face
# on this machine to expose it.
import insightface as _insightface

_INSIGHTFACE_OBJECTS_DIR = Path(_insightface.__file__).resolve().parent / "data" / "objects"
if _INSIGHTFACE_OBJECTS_DIR.is_dir():
    datas.append((str(_INSIGHTFACE_OBJECTS_DIR), "objects"))
    print(f"[sentra.spec] bundling insightface reference objects from {_INSIGHTFACE_OBJECTS_DIR}")
else:
    raise SystemExit(
        f"\nERROR: insightface is installed but its 'objects' data directory is not at:\n"
        f"    {_INSIGHTFACE_OBJECTS_DIR}\n\n"
        "Every detected face would fail recognition at runtime with "
        "\"'NoneType' object has no attribute 'shape'\" deep inside insightface.\n"
    )

# onvif-zeep keeps its WSDL files in a TOP-LEVEL `wsdl/` directory beside the
# package rather than inside it, so collect_data_files("onvif") above does not
# see them. Without this, ONVIF camera discovery fails only at the moment
# someone tries to use it, with a file-not-found from deep inside zeep.
# sentra_paths.onvif_wsdl_dir() is the other half: it points the library here.
try:
    import onvif as _onvif

    _WSDL_DIR = Path(_onvif.__file__).resolve().parent.parent / "wsdl"
    if _WSDL_DIR.is_dir():
        datas.append((str(_WSDL_DIR), "wsdl"))
        print(f"[sentra.spec] bundling ONVIF WSDLs from {_WSDL_DIR}")
    else:
        raise SystemExit(
            f"\nERROR: onvif-zeep is installed but its WSDL directory is not at:\n"
            f"    {_WSDL_DIR}\n\n"
            "ONVIF camera discovery would fail at runtime with a file-not-found\n"
            "from inside zeep. Locate the 'wsdl' directory in site-packages and\n"
            "point this at it.\n"
        )
except ImportError:
    raise SystemExit(
        "\nERROR: onvif-zeep is not installed in the build environment.\n\n"
        "ONVIF camera discovery is offered in the Cameras tab, so a build\n"
        "without it ships a feature that cannot work.\n"
        "    pip install -r backend_v2/requirements.txt\n"
    )

# --- Hidden imports -------------------------------------------------------
# The app modules are imported through sys.path manipulation at runtime rather
# than by literal top-level import statements, so PyInstaller's static analysis
# cannot see them. They must be named explicitly.

hiddenimports = [
    # Sentra's own modules
    "sentra_paths",
    "sentra_version",
    "updater",
    "data_pack",
    "camera_config",
    "camera_store",
    "onvif_discovery",
    "event_logger",
    "anomaly_detection",
    "face_recognition",
    "face_register",
    "visitor_store",
    "main",
    "database",
    "people_store",
    "visitor_service",
    # uvicorn resolves these by string name at startup
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # The update check talks HTTPS. Named explicitly because a frozen build
    # that omits the SSL machinery still starts, still serves the dashboard,
    # and only fails at the moment it tries to reach the update feed — exactly
    # the kind of silent, late failure this project keeps getting bitten by.
    "ssl",
    "_ssl",
    "hashlib",
    "_hashlib",
    # onnxruntime picks its provider dynamically
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_pybind11_state",
]

# insightface loads its model classes reflectively from model_zoo.
hiddenimports += collect_submodules("insightface")
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("skimage")
# zeep resolves SOAP transports and XML plugins by name at runtime, and onvif
# is imported lazily inside a function so static analysis never sees it.
hiddenimports += collect_submodules("zeep")
hiddenimports += ["onvif", "onvif.client", "onvif.exceptions"]
hiddenimports += _scipy_hidden + _skimage_hidden

binaries = list(_scipy_binaries) + list(_skimage_binaries)

# --- Analysis -------------------------------------------------------------

a = Analysis(
    [str(PROJECT_ROOT / "sentra_app.py")],
    pathex=[
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "Formal_Code"),
        str(PROJECT_ROOT / "backend_v2"),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Excluded to keep the installer down. This list is short on purpose.
    #
    # matplotlib, torch.testing and torch.distributions were all on it and all
    # look like dead weight for an app that renders no plots and trains no
    # models. ultralytics imports every one of them, so excluding them made the
    # pose model fail to load — and because that failure degrades gracefully by
    # design, the app still started and served the dashboard with fight
    # detection silently off. Caught on the macOS build (testable locally); the
    # identical exclusions were live here. Each was removed only after
    # measuring sys.modules after an actual pose inference.
    #
    # Before adding anything here: run `Sentra.exe --selftest` on the built
    # bundle. The CI workflow does this and fails on a non-zero exit, which is
    # what stops this mistake shipping again.
    #
    # polars (added 2026-08-02, ~184MB via its native `_polars_runtime_32`
    # backend): ultralytics depends on it, but every reference in its source
    # is a function-local import inside training-results CSV export and
    # results plotting — never inference. Verified, not assumed, following
    # the same rule as the matplotlib/torch.testing case above: blocked
    # `import polars` outright and ran Sentra's actual FightDetector pipeline
    # (load_pose_model, detect_people, analyze_frame — the real
    # anomaly_detection.py code path, not a synthetic call) end to end with
    # no error. Sentra never trains a model, so the only code that needs
    # polars is code Sentra never reaches.
    excludes=[
        "tkinter",
        "PyQt5",
        "PySide2",
        "notebook",
        "IPython",
        "pandas",
        "polars",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Sentra",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX mangles onnxruntime/torch DLLs — do not enable
    # Windowed for the client; logs go to .run_logs rather than a console.
    # Set SENTRA_CONSOLE=1 before building to get a console window instead —
    # worth doing for the first build on the Windows test machine, because a
    # windowed build that dies during startup shows the user nothing at all.
    console=bool(os.environ.get("SENTRA_CONSOLE")),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "windows" / "sentra.ico")
    if (PROJECT_ROOT / "windows" / "sentra.ico").is_file()
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Sentra",
)
