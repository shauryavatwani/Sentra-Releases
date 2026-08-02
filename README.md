# Sentra

B2B AI CCTV intelligence for institutional security — face recognition, fight
detection, and visitor management, running against real camera footage.

## Download

### ➡️ [**Get the latest release**](../../releases/latest)

Grab `SentraSetup-*.exe` for **Windows**, or `Sentra-*.dmg` for **macOS**
(Apple Silicon). Run it, click through, done — everything is bundled, so
there is no Python to install and nothing to configure.

**Your computer will warn you the first time.** Neither build is signed with
a paid certificate, so:

- **Windows** — SmartScreen shows *"Windows protected your PC"*.
  Click **More info → Run anyway**.
- **macOS** — Gatekeeper says it *"cannot be opened"*. **Right-click** the
  app and choose **Open** (double-clicking will not offer the option).

Both are expected, and both only happen once.

After that Sentra checks for updates by itself and offers them from
**Settings**, so nobody needs to come back here to stay current.

Once installed, Sentra checks for updates on its own and offers them from
**Settings** — a security team never has to reinstall by hand.

### Moving an existing setup to a new machine

A fresh install has nobody registered — the installer deliberately carries no
photographs or face data (see [Why the repo looks empty of data](#why-the-repo-looks-empty-of-data)
below). To bring over an existing roster and detection history, use
**Settings → Export data pack** on the source machine, then
**Register person → Import a data pack** on the new one.

## Repository layout

```
Formal_Code/        the AI engine — face recognition, fight detection, camera
                     management, visitor passes. Runs as its own process.
backend_v2/          the web backend (FastAPI) and the dashboard it serves.
tests/                unit tests for the fight-detection scoring logic.

windows/             PyInstaller spec + Inno Setup script -> SentraSetup.exe
macos/               PyInstaller spec + build script -> Sentra.app / .dmg
.github/workflows/  CI: builds both installers on a tag push (vX.Y.Z)

sentra_app.py        the packaged app's single entry point (both the web
                     server and the camera engine run from this one binary)
```

## Building from source

You only need this if you're changing the code. See
[`WINDOWS_BUILD.md`](WINDOWS_BUILD.md) for the Windows build (requires a
Windows machine — PyInstaller cannot cross-compile) or run
`bash macos/build_macos.sh` on a Mac.

To publish a new version: bump `VERSION` in
[`Formal_Code/sentra_version.py`](Formal_Code/sentra_version.py), then
`git tag vX.Y.Z && git push origin vX.Y.Z`. CI builds both platforms and
attaches them to a new GitHub Release automatically.

## Why the repo looks empty of data

This repo is public — the auto-updater needs to reach it without
credentials — and Sentra's own data is photographs, face embeddings, and a
log of who was seen where. `Faces/`, `Visitors/`, and `Database/` are
gitignored on purpose; nothing here identifies a real person. A first install
seeds itself empty, and real deployments move data between machines using the
export/import flow above, never through this repository.
