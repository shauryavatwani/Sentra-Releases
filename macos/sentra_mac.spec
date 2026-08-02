# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Sentra (macOS).

Run from the project root:

    pyinstaller macos/sentra_mac.spec --noconfirm

Produces dist/Sentra.app. macos/build_macos.sh wraps that into a .dmg.

Differences from the Windows spec that matter:

* Output is a **.app bundle**, not a folder — that is what "an application" is
  on macOS, and it is what can be dragged into /Applications.
* ``console`` is meaningless here; a .app never has a terminal attached, so the
  stdout/stderr-is-None handling in sentra_app.py matters just as much as it
  does on Windows.
* The bundle is **ad-hoc signed**. On Apple Silicon an unsigned binary is
  killed by the kernel outright rather than merely warned about, so this is not
  optional the way it is on Intel.
"""

import datetime
import json
import platform
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH).resolve().parent

# --- Version stamping -----------------------------------------------------
# Same single source of truth as the Windows build: the literal lives in
# Formal_Code/sentra_version.py and is read here by regex (so this spec needs
# none of Sentra's runtime dependencies importable to run).

_version_src = (PROJECT_ROOT / "Formal_Code" / "sentra_version.py").read_text(encoding="utf-8")
_match = re.search(r'^VERSION\s*=\s*"([^"]+)"', _version_src, re.MULTILINE)
if not _match:
    raise SystemExit(
        "\nERROR: could not read VERSION from Formal_Code/sentra_version.py.\n"
    )
APP_VERSION = _match.group(1)
BUILD_DATE = datetime.date.today().isoformat()

BUILD_INFO = PROJECT_ROOT / "macos" / "build_info.json"
BUILD_INFO.write_text(
    json.dumps(
        {
            "version": APP_VERSION,
            "build_date": BUILD_DATE,
            "channel": "release",
            "platform": f"macOS-{platform.machine()}",
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"[sentra_mac.spec] Building Sentra {APP_VERSION} for macOS-{platform.machine()}")

# --- Model files ----------------------------------------------------------
# The two things the app cannot download at the client's site. A build that
# silently omits them produces an app that looks fine and then fails on first
# use, so fail the build here instead.

INSIGHTFACE_MODELS = Path.home() / ".insightface"
BUFFALO = INSIGHTFACE_MODELS / "models" / "buffalo_l"
if not BUFFALO.is_dir():
    raise SystemExit(
        f"\nERROR: InsightFace model set not found at:\n    {BUFFALO}\n\n"
        "Run the app once on this machine to download it, or copy the\n"
        ".insightface folder in from another machine.\n"
    )

POSE_WEIGHTS = PROJECT_ROOT / "Formal_Code" / "yolov8n-pose.pt"
if not POSE_WEIGHTS.is_file():
    raise SystemExit(
        f"\nERROR: pose model not found at:\n    {POSE_WEIGHTS}\n\n"
        "Without it fight detection silently disables itself.\n"
    )

# The face --selftest recognises to prove this build works. A missing fixture
# fails the build rather than quietly weakening the only gate standing between
# a silently-broken bundle and a customer's machine.
SELFTEST_FACE = PROJECT_ROOT / "tests" / "fixtures" / "selftest_face.jpg"
if not SELFTEST_FACE.is_file():
    raise SystemExit(
        f"\nERROR: self-test face image not found at:\n    {SELFTEST_FACE}\n\n"
        "--selftest needs it to prove face recognition actually works in the\n"
        "built bundle. Without it the build ships unverified.\n"
    )

# --- Bundled data ---------------------------------------------------------

datas = [
    (str(PROJECT_ROOT / "backend_v2" / "dashboard.html"), "backend_v2"),
    (str(PROJECT_ROOT / "backend_v2" / "login.html"), "backend_v2"),
    (str(POSE_WEIGHTS), "Formal_Code"),
    (str(BUFFALO), "insightface/models/buffalo_l"),
    (str(BUILD_INFO), "."),
    # 14KB synthetic face --selftest runs through the full recognition
    # pipeline. Bundled deliberately: every check that did not put a real face
    # through model.get() passed on builds that could not recognise anyone.
    (str(SELFTEST_FACE), "tests/fixtures"),
]

datas += collect_data_files("ultralytics")
datas += collect_data_files("insightface")
datas += collect_data_files("zeep")
datas += collect_data_files("onvif")

# scipy + scikit-image. InsightFace's face alignment (face_align.norm_crop,
# called for every detected face) reaches skimage.transform -> scipy, and a
# bundle missing scipy's compiled extensions recognises nobody while looking
# completely healthy otherwise. The macOS bundle has happened to pick these up
# already, but that was incidental rather than declared — and the equivalent
# gap is what broke the Windows build. Collected explicitly on both platforms
# so neither depends on a hook version behaving a particular way.
_scipy_datas, _scipy_binaries, _scipy_hidden = collect_all("scipy")
_skimage_datas, _skimage_binaries, _skimage_hidden = collect_all("skimage")
datas += _scipy_datas + _skimage_datas

# insightface.data.get_object() reads meanshape_68.pkl (used by the
# landmark_3d_68 model every detected face goes through) from a TOP-LEVEL
# `objects/` directory beside the bundle root when frozen, not from inside the
# insightface package — the same sibling-directory trap as onvif-zeep's WSDLs
# below. get_object() swallows the missing file and returns None, so the
# failure surfaces three calls later as `'NoneType' object has no attribute
# 'shape'` inside estimate_affine_matrix_3d23d, for every detected face, on
# every frozen build on both platforms — "the live feed works but nobody is
# ever recognised". This was never exercised on a real face on this machine
# before (the packaged app's camera never connected), so it shipped in 1.0.3
# on both platforms undetected. Found 2026-08-02 by tracing the traceback a
# real Windows install produced.
import insightface as _insightface

_INSIGHTFACE_OBJECTS_DIR = Path(_insightface.__file__).resolve().parent / "data" / "objects"
if not _INSIGHTFACE_OBJECTS_DIR.is_dir():
    raise SystemExit(
        f"\nERROR: insightface is installed but its 'objects' data directory is not at:\n"
        f"    {_INSIGHTFACE_OBJECTS_DIR}\n"
    )
datas.append((str(_INSIGHTFACE_OBJECTS_DIR), "objects"))

# onvif-zeep keeps its WSDLs in a TOP-LEVEL `wsdl/` directory beside the
# package rather than inside it, so collect_data_files does not see them.
# sentra_paths.onvif_wsdl_dir() is the other half, pointing the library here.
try:
    import onvif as _onvif

    _WSDL_DIR = Path(_onvif.__file__).resolve().parent.parent / "wsdl"
    if not _WSDL_DIR.is_dir():
        raise SystemExit(
            f"\nERROR: onvif-zeep is installed but its WSDLs are not at:\n    {_WSDL_DIR}\n"
        )
    datas.append((str(_WSDL_DIR), "wsdl"))
except ImportError:
    raise SystemExit(
        "\nERROR: onvif-zeep is not installed.\n"
        "    pip install -r backend_v2/requirements.txt\n"
    )

# --- Hidden imports -------------------------------------------------------
# Sentra's own modules are reached through sys.path manipulation rather than
# literal imports, so PyInstaller's static analysis cannot see them.

hiddenimports = [
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
    # The update check talks HTTPS; a bundle missing the SSL machinery still
    # starts and only fails when it reaches the update feed.
    "ssl",
    "_ssl",
    "hashlib",
    "_hashlib",
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_pybind11_state",
]

hiddenimports += collect_submodules("insightface")
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("skimage")
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
    # Excluded to keep the bundle down. This list is short on purpose.
    #
    # matplotlib, torch.testing and torch.distributions were all on it and all
    # look like dead weight for an app that renders no plots and trains no
    # models. ultralytics imports every one of them, so excluding them made the
    # pose model fail to load — and because that failure degrades gracefully by
    # design, the app still started and served the dashboard with fight
    # detection silently off. Each was removed only after measuring
    # sys.modules after an actual pose inference, not by reasoning about it.
    #
    # Before adding anything here: run `Sentra.app/Contents/MacOS/Sentra
    # --selftest` on the built bundle. The build script does this and fails on
    # a non-zero exit, which is what stops this mistake shipping again.
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
    upx=False,  # UPX mangles onnxruntime/torch dylibs — do not enable
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # host architecture; see build_macos.sh for why not universal2
    codesign_identity=None,  # ad-hoc; see BUNDLE note below
    entitlements_file=None,
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

# --- The .app bundle ------------------------------------------------------

app = BUNDLE(
    coll,
    name="Sentra.app",
    icon=str(PROJECT_ROOT / "macos" / "Sentra.icns")
    if (PROJECT_ROOT / "macos" / "Sentra.icns").is_file()
    else None,
    bundle_identifier="com.dpsbangaloreeast.sentra",
    version=APP_VERSION,
    info_plist={
        "CFBundleName": "Sentra",
        "CFBundleDisplayName": "Sentra",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "NSHumanReadableCopyright": "Delhi Public School Bangalore East",
        # Sentra's window is the dashboard, which opens in the default browser.
        # Without this the app would sit in the Dock as a second, empty
        # application the user cannot interact with.
        "LSUIElement": True,
        "LSMinimumSystemVersion": "11.0",
        # The dashboard is served over plain HTTP on localhost. ATS would
        # otherwise block the app's own requests to it.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        # macOS asks the user before letting an app reach devices on the LAN.
        # Cameras are on the LAN, so without a reason string the prompt is
        # blank and reads as suspicious.
        "NSLocalNetworkUsageDescription":
            "Sentra connects to your security cameras on this network.",
        # Only relevant if a camera is ever attached directly rather than over
        # RTSP, but a missing string is an immediate crash if that ever happens.
        "NSCameraUsageDescription":
            "Sentra uses connected cameras for security monitoring.",
    },
)
