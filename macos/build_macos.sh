#!/bin/bash
# =============================================================================
#  Sentra — macOS build
#
#  Run from the PROJECT ROOT:
#      bash macos/build_macos.sh
#
#  Produces:
#      dist/Sentra.app                     the application itself
#      macos/output/Sentra-<version>.dmg   what you hand to someone else
#
#  Unlike the Windows build, this one can run on the development machine, so
#  the app can be launched and checked before it is given to anyone.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
echo "=== Sentra macOS build ==="
echo "Project root: $PROJECT_ROOT"
echo

# --- Interpreter ------------------------------------------------------------
# Everything must be built with the same interpreter the app is tested under.
# This project has two Python environments that are NOT interchangeable (numpy
# 1.x vs 2.x); building under the wrong one produces a bundle whose pickles the
# engine cannot read.
PYBIN="${SENTRA_PYTHON:-/opt/anaconda3/envs/sharktank/bin/python3}"
if [ ! -x "$PYBIN" ]; then
    echo "ERROR: interpreter not found: $PYBIN"
    echo "Set SENTRA_PYTHON to the Python that has Sentra's dependencies."
    exit 1
fi
echo "Interpreter: $PYBIN"
"$PYBIN" -c "import sys; print('  ->', sys.version.split()[0], sys.platform)"
ARCH="$(uname -m)"
echo "  -> architecture: $ARCH"
echo

# --- Dependency gate --------------------------------------------------------
# Checked up front rather than 10 minutes into a PyInstaller run.
echo "[1/5] Checking dependencies..."
"$PYBIN" - <<'PYCHECK'
import sys
missing = []
for mod in ("PyInstaller", "fastapi", "uvicorn", "cv2", "insightface",
            "onnxruntime", "numpy", "torch", "ultralytics", "onvif", "zeep"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print("  ERROR missing:", ", ".join(missing))
    print("  pip install -r backend_v2/requirements.txt && pip install pyinstaller==6.11.1")
    sys.exit(1)
import numpy
if numpy.__version__.startswith("2."):
    print(f"  ERROR numpy is {numpy.__version__}; InsightFace needs <2. See requirements.txt.")
    sys.exit(1)
print("  all present, numpy", numpy.__version__)
PYCHECK
echo

# --- Build ------------------------------------------------------------------
echo "[2/5] Building Sentra.app (several minutes)..."
rm -rf build/Sentra dist/Sentra dist/Sentra.app
"$PYBIN" -m PyInstaller macos/sentra_mac.spec --noconfirm --clean
if [ ! -d "dist/Sentra.app" ]; then
    echo "ERROR: dist/Sentra.app was not produced."
    exit 1
fi
echo "  built: dist/Sentra.app"
echo

# --- Signing ----------------------------------------------------------------
# On Apple Silicon an unsigned or broken-signature binary is killed by the
# kernel on launch, not merely warned about — so ad-hoc signing is required for
# the app to start at all, even locally. This is NOT the same as a Developer ID
# signature: Gatekeeper will still warn a user who downloads it (see README).
echo "[3/5] Ad-hoc signing..."
codesign --force --deep --sign - dist/Sentra.app 2>&1 | sed 's/^/  /' || true
if codesign --verify --deep --strict dist/Sentra.app 2>/dev/null; then
    echo "  signature verifies"
else
    echo "  WARNING: signature did not verify; the app may refuse to launch."
fi
echo

# --- Self-test --------------------------------------------------------------
# Runs INSIDE the bundle that was just built, so it tests what will actually
# ship rather than the development environment. This exists because a bundle
# missing torch.testing loaded, served the dashboard, and had fight detection
# silently switched off — a build that looked completely successful.
echo "[3.5/5] Self-test inside the bundle..."
if ! ./dist/Sentra.app/Contents/MacOS/Sentra --selftest; then
    echo
    echo "ERROR: the built app failed its own self-test (see above)."
    echo "Do NOT ship this build — a capability is missing from the bundle."
    exit 1
fi
echo

# --- Version ----------------------------------------------------------------
VERSION="$("$PYBIN" -c "
import re,pathlib
src=pathlib.Path('Formal_Code/sentra_version.py').read_text(encoding='utf-8')
print(re.search(r'^VERSION\s*=\s*\"([^\"]+)\"', src, re.M).group(1))
")"
echo "[4/5] Version: $VERSION"
APP_SIZE="$(du -sh dist/Sentra.app | cut -f1)"
echo "  app size: $APP_SIZE"
echo

# --- DMG --------------------------------------------------------------------
# A .dmg with an /Applications symlink is the drag-to-install convention every
# Mac user already knows. hdiutil ships with macOS, so no extra tooling.
echo "[5/5] Building the disk image..."
mkdir -p macos/output
DMG="macos/output/Sentra-$VERSION.dmg"
rm -f "$DMG"

STAGE="$(mktemp -d)"
cp -R dist/Sentra.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/READ ME FIRST.txt" <<EOF
Sentra $VERSION
${VERSION//?/=}=======

TO INSTALL
  Drag Sentra onto the Applications folder shown beside it.

THE FIRST TIME YOU OPEN IT
  macOS will say Sentra "cannot be opened because Apple cannot check it for
  malicious software". That is because this build is not signed with a paid
  Apple Developer certificate — not because anything is wrong with it.

  To open it anyway:
      Right-click (or Control-click) Sentra in Applications, choose Open,
      then click Open in the dialog.

  You only have to do this once. Afterwards it opens normally.

WHAT HAPPENS ON LAUNCH
  Sentra starts in the background and opens its dashboard in your browser at
      http://localhost:8000
  There is no separate window — the dashboard IS the application.

  Sign in with the credentials shown on the login page.

TO ADD YOUR EXISTING PEOPLE AND HISTORY
  Register person -> Import a data pack -> choose the .sentra file you were
  given. Everyone is restored in one step.

TO QUIT
  Open Activity Monitor, search for "Sentra", and quit both entries.

YOUR DATA LIVES HERE
  ~/Library/Application Support/Sentra
  Updating or reinstalling Sentra never touches it.
EOF

hdiutil create -volname "Sentra $VERSION" \
    -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"

DMG_SIZE="$(du -sh "$DMG" | cut -f1)"
echo "  $DMG ($DMG_SIZE)"
echo
echo "============================================================"
echo "  BUILD COMPLETE"
echo "    App : dist/Sentra.app        ($APP_SIZE)"
echo "    DMG : $DMG  ($DMG_SIZE)"
echo
echo "  The DMG is the file to send. It is UNSIGNED, so the first"
echo "  launch needs right-click -> Open (explained in the DMG's"
echo "  READ ME FIRST.txt)."
echo "============================================================"
