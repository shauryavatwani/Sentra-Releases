# Building Sentra for Windows

This produces `SentraSetup.exe` — a normal Windows installer that puts Sentra in
Program Files with Start Menu and desktop shortcuts. The client double-clicks
it, clicks Next a few times, and gets a working app. No Python, no terminal, no
internet needed on their machine.

**You must build on a Windows machine.** PyInstaller cannot cross-compile — a
Windows `.exe` can only be produced on Windows. Everything in this repo is
ready; the build itself just has to run there.

---

## 1. One-time setup on the build machine

| Requirement | Notes |
|---|---|
| **Windows 10/11, 64-bit** | |
| **Python 3.12** (3.10/3.11 also fine) | From [python.org](https://www.python.org/downloads/) — tick **"Add Python to PATH"** during install. **Not 3.13+** (see below). |
| **Inno Setup 6** | From [jrsoftware.org/isdl.php](https://jrsoftware.org/isdl.php). Default install location is detected automatically. |
| **~15 GB free disk** | The build tree is large before compression. |
| **Internet** | Only for the build machine, only during the build. |

> **Why not Python 3.13:** numpy is pinned to 1.26.4, which has no 3.13 wheels.
> On 3.13 pip tries to compile numpy from source and fails with a wall of
> compiler errors. The build script checks this and stops with a clear message.

## 2. Copy the project across

Copy the whole `Shark_Tank` folder to the Windows machine. You can skip these —
they're large and not needed for the build:

```
Human Faces Dataset/   archive.zip   specs/   .run_logs/   backend/
```

> `backend/` is the retired original UI. Only `backend_v2/` ships.

## 3. Build

Open **Command Prompt**, `cd` into the project folder, and run:

```bat
windows\build_windows.bat
```

Takes roughly 15–30 minutes on the first run (torch and the InsightFace models
are the slow parts) and only a few minutes afterwards. Output:

```
windows\output\SentraSetup.exe     <- give this to the client
dist\Sentra\                       <- the raw unpacked app, for testing
```

### Build the first one with a console

A windowed build that crashes on startup shows the user *nothing*. For your
first test build, get a console window so you can read the traceback:

```bat
set SENTRA_CONSOLE=1
windows\build_windows.bat
```

Once it runs cleanly, rebuild without that variable (open a fresh Command
Prompt) for the copy you actually hand over.

---

## 4. Test before handing it over

Install it on the build machine, then check each of these — in this order,
because each one depends on the previous working:

1. **It launches.** Start Menu → Sentra. A browser opens on
   `http://localhost:8000` showing the login page.
2. **Login works** — `sharktanktest` / `demo`.
3. **Fight detection loaded.** Open
   `C:\ProgramData\Sentra\.run_logs\face_recognition.log` and look for
   `Pose model loaded`. If it says `Fight detection disabled`, ultralytics or
   torch didn't make it into the bundle — that's the single most likely
   packaging failure, and it is silent in the UI.
4. **Camera.** Cameras tab → set the RTSP URL for the client's camera → then
   **Restart engine** in the top bar. Live Monitor should show video.
5. **Registration writes.** Register a person, then confirm
   `C:\ProgramData\Sentra\Database\face_embeddings.pkl` updated. This is the
   check that the read-only-Program-Files problem is actually solved.
6. **Restart button.** Click it; the log should show the engine stopping and
   starting, with only one `Sentra.exe --engine` process in Task Manager
   afterwards.
7. **From another device.** Open `http://<pc-ip>:8000` from a phone on the same
   Wi-Fi. If this fails but localhost works, the firewall rule didn't apply.

---

## Where things live once installed

```
C:\Program Files\Sentra\          the app itself (read-only)
  Sentra.exe

C:\ProgramData\Sentra\            everything the app writes
  Database\detections.db          detection + alert history
  Database\face_embeddings.pkl    registered faces
  Database\camera_config.json     cameras (name, RTSP URL, AI on/off)
  Database\update_config.json     optional: where to check for updates
  Faces\                          uploaded registration photos
  Visitors\                       temporary-pass gate photos
  .run_logs\                      engine and server logs
  .updates\                       staged installer, deleted after it applies
```

The split matters: Program Files is read-only for standard Windows users, so
anything the app saves has to live under ProgramData. Uninstalling deliberately
**keeps** `Database\`, `Faces\` and `Visitors\` — it will not destroy the
client's recorded history, the faces they registered, or the visitor log.

To put the data on another drive, set a system environment variable
`SENTRA_DATA_DIR` to the folder you want.

### What a first install ships with

A **fresh** install seeds `Database\`, `Faces\` and `Visitors\` from this
project, so the client receives the complete working system — existing
detection history, registered people, and visitor records included.

An **upgrade** seeds none of it. That is decided once before any file is
copied (`InitializeSetup` in `windows/sentra.iss`, keyed on whether
`detections.db` already exists), so seed data can never reappear over the top
of the client's own work months later.

---

## Releasing an update

The version number lives in exactly one place: `VERSION` in
`Formal_Code/sentra_version.py`. The PyInstaller spec reads it and writes both
`windows/build_info.json` (what Settings → About shows) and
`windows/version_define.iss` (what the installer stamps), so those three can
never disagree.

```
1. Bump VERSION in Formal_Code/sentra_version.py
2. windows\build_windows.bat
3. Upload windows\output\{SentraSetup-<ver>.exe, version.json, release_notes.md}
   to the same folder — see windows\output\PUBLISHING.txt
```

Clients check `version.json` automatically ~20s after each launch, and on
demand from Settings → Software updates. Nothing installs by itself: the
operator downloads and applies it explicitly, and only an **admin** account
can — a guard sees the notice but no install button, enforced server-side.

**The checksum is not optional.** `version.json` carries a `sha256` and the
client refuses any download that does not match it, before *and* again
immediately before running it. This is a binary that gets executed with
administrator rights, so an unverified download would let anyone who can spoof
the feed host run code on every client machine. `make_release.py` computes the
digest from the file it just built, so re-upload the installer and manifest
**together** — a new installer with a stale manifest is rejected by every
client.

Feeds must be `https://`. Plain HTTP is refused outright.

### Moving to a different host

No rebuild needed. The updater only ever fetches a JSON file over HTTPS, so
GitHub Releases, S3, Cloudflare R2, Azure Blob and a plain nginx box are all
interchangeable. Point a client at a new one with either:

```
C:\ProgramData\Sentra\Database\update_config.json
  { "feed_url": "https://your-host/version.json" }
```

or the `SENTRA_UPDATE_URL` environment variable, which wins over the file.
Set `SENTRA_DISABLE_UPDATE_CHECK=1` to switch the automatic check off entirely.

---

## Troubleshooting

**`ModuleNotFoundError` for `main`, `sentra_paths`, `camera_config`… at startup**
Those modules are imported via `sys.path` manipulation, so PyInstaller can't
see them statically. They're listed in `hiddenimports` in `windows/sentra.spec`
— add any new module there too.

**"Fight detection disabled" in the log**
ultralytics/torch didn't get bundled. Check the PyInstaller warnings at
`build/Sentra/warn-Sentra.txt`. Face recognition keeps working without it, so
nothing in the UI tells you — always check the log.

**App starts, dashboard is blank or 404**
`dashboard.html` / `login.html` weren't bundled. Confirm they're present in
`dist\Sentra\_internal\backend_v2\`.

**InsightFace error about missing models**
The `buffalo_l` set didn't get included. Verify
`dist\Sentra\_internal\insightface\models\buffalo_l\` contains the `.onnx`
files. The spec deliberately fails the build if they're missing on the build
machine — if you skipped that error, this is the consequence.

**Installer built but the app won't start on the client's PC**
Most likely a missing Visual C++ runtime. Install the
[VC++ 2015–2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
on the client machine.

**Antivirus flags the installer**
Expected for unsigned PyInstaller executables. The real fix is a code-signing
certificate. Without one, the client's IT may need to whitelist it.

---

## Changing the port

Port 8000 is set in three places and all three must agree:

- `sentra_app.py` — `APP_PORT`
- `Formal_Code/face_recognition.py` — `DASHBOARD_WS_URLS`
- `windows/sentra.iss` — the two `netsh` firewall lines
