"""Update checking and staging for Sentra.

Deliberately built around one idea: **an update feed is just a JSON file at a
URL.** Nothing here knows what a GitHub Release, an S3 bucket or a Cloudflare R2
object is, so moving between them is a URL change in
``Database/update_config.json`` and never a code change.

Manifest schema (host this as ``version.json`` next to the installers)::

    {
      "version":    "1.1.0",              # required
      "url":        "https://.../SentraSetup-1.1.0.exe",   # Windows payload —
      "sha256":     "9f86d081884c7d...",  # kept flat for backward compat with
      "size":       734003200,            # already-installed 1.0.2/1.0.3
                                           # clients, whose updater only ever
                                           # reads these three flat fields.
      "build_date": "2026-08-15",         # optional, shown in the UI
      "notes":      "Fixed the camera reconnect loop.",     # optional
      "notes_url":  "https://.../release_notes.md",         # optional
      "mandatory":  false,                # optional, styles the prompt louder
      "platforms": {                      # one entry per platform this
        "win32":  {"url": "...", "sha256": "...", "size": ...},  # version's
        "darwin": {"url": "...", "sha256": "...", "size": ...}   # updater
      }                                                            # reads
    }

``sha256`` is **required and enforced** for whichever platform's entry is
actually used. This module downloads an installer/disk image and then runs or
replaces itself with it, so an unverified download would hand anyone who can
spoof the feed host a way to run code on every client machine. An entry
without a valid digest is rejected rather than trusted, and HTTP feeds are
refused outright — the check is only as trustworthy as its transport. See
:func:`_platform_entry` for exactly how a platform's entry is chosen.

Nothing is ever installed automatically. The check runs by itself, the download
runs when the operator asks, and applying it is always an explicit click.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import sentra_paths
import sentra_version

# --- Tunables ---------------------------------------------------------------

CHECK_TIMEOUT_SECONDS = 12
DOWNLOAD_TIMEOUT_SECONDS = 30
# How long a check result stays fresh. The startup check and any manual click
# inside this window reuse the cached answer instead of hitting the network.
CACHE_TTL_SECONDS = 6 * 60 * 60
# Delay before the automatic startup check, so it never competes with the model
# load and camera connect that happen in the same first seconds.
STARTUP_CHECK_DELAY_SECONDS = 20

STAGING_DIR = sentra_paths.DATA_ROOT / ".updates"
USER_AGENT = f"Sentra/{sentra_version.VERSION}"

# --- State ------------------------------------------------------------------
# One process-wide state object, guarded by a lock, snapshot-copied on read.
# The UI polls this; the worker threads mutate it.

_lock = threading.RLock()
_state: dict[str, Any] = {
    # idle | checking | up_to_date | available | downloading | ready | error
    "status": "idle",
    "current_version": sentra_version.VERSION,
    "latest_version": None,
    "notes": "",
    "notes_url": "",
    "release_date": "",
    "mandatory": False,
    "size": 0,
    "downloaded": 0,
    "progress": 0.0,
    "staged_path": "",
    "error": "",
    "last_checked": 0.0,
    "feed_url": "",
    "supported": sys.platform in ("win32", "darwin"),
}
_download_thread: threading.Thread | None = None


def _set(**fields: Any) -> None:
    with _lock:
        _state.update(fields)


def state() -> dict:
    """A snapshot of the current update state, safe to serialise.

    Underscore-prefixed keys are this module's own bookkeeping (the resolved
    download URL and its expected digest) and are stripped rather than sent to
    the browser — the state object is an API response, not an internal dump.
    """
    with _lock:
        snapshot = {k: v for k, v in _state.items() if not k.startswith("_")}
    snapshot["last_checked_iso"] = (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(snapshot["last_checked"]))
        if snapshot["last_checked"]
        else ""
    )
    return snapshot


# --- Checking ---------------------------------------------------------------


def _fetch_manifest(url: str) -> dict:
    if not url.lower().startswith("https://"):
        # An update feed decides which executable gets run as administrator on
        # this machine. Plain HTTP lets anyone on the path choose that.
        raise ValueError("The update feed must be an https:// URL.")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=CHECK_TIMEOUT_SECONDS) as response:
        # Cap the read: a manifest is a few hundred bytes, and a feed host that
        # answers with something enormous should not exhaust memory here.
        raw = response.read(256 * 1024)
    return json.loads(raw.decode("utf-8"))


def _installer_filename(version: str) -> str:
    """The staged file name for this platform's payload.

    Matches what each build actually produces: windows/make_release.py stamps
    the installer as SentraSetup-<version>.exe, macos/build_macos.sh names the
    disk image Sentra-<version>.dmg directly. Anything else (a platform that
    isn't win32/darwin) never reaches this — state()["supported"] is False.
    """
    return f"SentraSetup-{version}.exe" if sys.platform == "win32" else f"Sentra-{version}.dmg"


def _manifest_version(manifest: dict) -> str:
    version = str(manifest.get("version", "")).strip()
    if not version:
        raise ValueError("The update manifest is missing a version number.")
    return version


def _platform_entry(manifest: dict) -> tuple[str, str, int]:
    """Return (url, sha256, size) for THIS platform, or raise ValueError.

    The manifest carries one payload per platform under "platforms" (each
    with its own url/sha256/size, since the Windows installer and the macOS
    disk image are different files needing different digests) plus flat
    top-level url/sha256 fields kept as the Windows payload for backward
    compatibility — an already-installed 1.0.2/1.0.3 client's updater only
    ever reads those flat fields and has no idea "platforms" exists, so they
    must keep meaning exactly what they always meant.

    Deliberately separate from :func:`_manifest_version` and only called once
    a newer version is confirmed to exist — a client that is already
    up to date (or, as happens on this dev machine, running a source build
    newer than the latest published release) must never fail a routine check
    just because *this platform's* entry happens to be missing or malformed;
    that field is only meaningful once there is actually something to offer.

    A platform with no entry (or a manifest predating this format entirely)
    falls through to the flat fields, which is correct for win32 and simply
    unsupported for anything else — caught by `state()["supported"]` before
    this is ever reached for a truly unhandled platform.
    """
    platforms = manifest.get("platforms")
    entry = platforms.get(sys.platform) if isinstance(platforms, dict) else None
    if not isinstance(entry, dict):
        # The flat top-level fields are only ever a Windows payload (that's
        # what every manifest before this multi-platform format meant, and
        # what a manifest generator that regresses to the old shape would
        # still be producing). Falling back to them for any *other* platform
        # would silently hand a mac client a Windows .exe under its own
        # sha256, which then fails confusingly deep inside hdiutil rather
        # than with a clear "not available" — so only win32 gets the fallback.
        if sys.platform != "win32":
            raise ValueError(f"No update is available for this platform ({sys.platform}) yet.")
        entry = manifest

    url = str(entry.get("url", "")).strip()
    digest = str(entry.get("sha256", "")).strip().lower()
    size = int(entry.get("size") or 0)

    if not url.lower().startswith("https://"):
        raise ValueError("The update manifest's download URL must be https://.")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(
            "The update manifest has no valid sha256 checksum for this platform. "
            "Sentra will not install an update it cannot verify."
        )
    return url, digest, size


def check(force: bool = False) -> dict:
    """Ask the feed whether a newer version exists.

    Never raises: a failed check records the reason in the state and leaves the
    running app completely untouched. Losing the ability to *check* for an
    update is not a reason to disturb someone watching a camera feed.
    """
    with _lock:
        fresh = (time.time() - _state["last_checked"]) < CACHE_TTL_SECONDS
        settled = _state["status"] in ("up_to_date", "available", "ready")
        if not force and fresh and settled:
            return state()
        if _state["status"] in ("checking", "downloading"):
            return state()
        _state["status"] = "checking"
        _state["error"] = ""

    feed = sentra_version.update_feed_url()
    _set(feed_url=feed)

    try:
        manifest = _fetch_manifest(feed)
        version = _manifest_version(manifest)
    except urllib.error.HTTPError as exc:
        _set(
            status="error",
            error=f"The update server answered {exc.code}. Check the feed URL in Settings.",
            last_checked=time.time(),
        )
        return state()
    except (urllib.error.URLError, OSError) as exc:
        _set(
            status="error",
            error=f"Could not reach the update server: {exc}",
            last_checked=time.time(),
        )
        return state()
    except (ValueError, TypeError) as exc:
        _set(status="error", error=str(exc), last_checked=time.time())
        return state()

    newer = sentra_version.is_newer(version)
    _set(
        latest_version=version,
        notes=str(manifest.get("notes", "")),
        notes_url=str(manifest.get("notes_url", "")),
        release_date=str(manifest.get("build_date", "")),
        mandatory=bool(manifest.get("mandatory", False)),
        last_checked=time.time(),
        error="",
    )

    if not newer:
        _set(status="up_to_date", staged_path="", downloaded=0, progress=0.0)
        return state()

    # Only resolved once a newer version is confirmed — see _platform_entry's
    # docstring for why this must not run before the is_newer check above.
    try:
        url, digest, size = _platform_entry(manifest)
    except (ValueError, TypeError) as exc:
        _set(status="error", error=str(exc), last_checked=time.time())
        return state()

    _set(size=size)

    # A build already staged for this exact version survives a re-check, so an
    # operator who downloaded then navigated away is not made to download again.
    staged = _staged_installer(version)
    if staged and _verify(staged, digest):
        _set(status="ready", staged_path=str(staged), progress=1.0, downloaded=staged.stat().st_size)
    else:
        _set(status="available", staged_path="", downloaded=0, progress=0.0)

    with _lock:
        _state["_url"] = url
        _state["_sha256"] = digest
    return state()


# --- Downloading ------------------------------------------------------------


def _staged_installer(version: str) -> Path | None:
    candidate = STAGING_DIR / _installer_filename(version)
    return candidate if candidate.is_file() else None


def _verify(path: Path, expected: str) -> bool:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return False
    return digest.hexdigest() == expected


def _download_worker(url: str, version: str, expected_sha: str) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    target = STAGING_DIR / _installer_filename(version)
    # Download to a partial file and rename only after the checksum passes, so
    # an interrupted download can never be mistaken for a staged installer.
    partial = target.with_suffix(".part")

    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared:
                _set(size=declared)

            received = 0
            with open(partial, "wb") as handle:
                while True:
                    block = response.read(512 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    received += len(block)
                    total = _state.get("size") or declared
                    _set(
                        downloaded=received,
                        progress=(received / total) if total else 0.0,
                    )

        if not _verify(partial, expected_sha):
            partial.unlink(missing_ok=True)
            _set(
                status="error",
                progress=0.0,
                downloaded=0,
                error=(
                    "The downloaded update failed its integrity check and was discarded. "
                    "Nothing was installed."
                ),
            )
            return

        partial.replace(target)
        _prune_old_installers(keep=target)
        _set(
            status="ready",
            staged_path=str(target),
            progress=1.0,
            downloaded=target.stat().st_size,
            error="",
        )
    except (urllib.error.URLError, OSError) as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        _set(status="error", progress=0.0, downloaded=0, error=f"Download failed: {exc}")


def _prune_old_installers(keep: Path) -> None:
    """Staged installers are hundreds of MB to ~1GB each; leaving every past
    one is not polite. Both platforms' patterns are checked regardless of
    which one is current, in case a staged file from a platform change (a
    reused data directory, an old build) is still sitting there."""
    try:
        for pattern in ("SentraSetup-*.exe", "Sentra-*.dmg"):
            for leftover in STAGING_DIR.glob(pattern):
                if leftover != keep:
                    leftover.unlink(missing_ok=True)
    except OSError:
        pass


def download() -> dict:
    """Start fetching the staged installer in the background."""
    global _download_thread

    with _lock:
        status = _state["status"]
        url = _state.get("_url", "")
        digest = _state.get("_sha256", "")
        version = _state.get("latest_version") or ""

        if status == "downloading":
            return state()
        if status == "ready":
            return state()
        if status != "available" or not url or not digest:
            return {**state(), "error": "No update is available to download. Check first."}

        _state["status"] = "downloading"
        _state["error"] = ""
        _state["downloaded"] = 0
        _state["progress"] = 0.0

    _download_thread = threading.Thread(
        target=_download_worker,
        args=(url, version, digest),
        name="sentra-update-download",
        daemon=True,
    )
    _download_thread.start()
    return state()


# --- Applying ---------------------------------------------------------------


def install() -> dict:
    """Verify the staged download, then hand off to the platform installer.

    On Windows the installer stops the running Sentra itself (``PrepareToInstall``
    in ``windows/sentra.iss``), replaces the program files, and leaves every
    folder under ``ProgramData\\Sentra`` exactly as it found it. On macOS this
    module does the equivalent itself — see :func:`_install_macos` — because
    there is no separate installer program to hand off to; the ``.dmg`` is just
    a disk image containing the new ``Sentra.app``.
    """
    with _lock:
        staged = _state.get("staged_path", "")
        digest = _state.get("_sha256", "")
        status = _state["status"]

    if status != "ready" or not staged:
        return {**state(), "error": "No update has been downloaded yet."}

    path = Path(staged)
    if not path.is_file():
        _set(status="available", staged_path="", error="The staged installer is missing.")
        return state()

    # Re-verify immediately before execution rather than trusting the check made
    # at download time — the file has been sitting on disk in a writable folder
    # since then, and this is the moment it becomes running code.
    if digest and not _verify(path, digest):
        path.unlink(missing_ok=True)
        _set(
            status="available",
            staged_path="",
            error="The staged installer no longer matches its checksum and was removed.",
        )
        return state()

    if sys.platform == "darwin":
        return _install_macos(path)
    if sys.platform != "win32":
        return {
            **state(),
            "error": "Automatic installation is not supported on this platform. "
            f"Run the downloaded installer manually: {path}",
        }

    # ShellExecuteW with the "runas" verb, NOT subprocess.Popen.
    #
    # The installer writes to Program Files and adds a firewall rule, so it is
    # manifested requireAdministrator. Sentra itself deliberately runs
    # unelevated (see `runasoriginaluser` in windows/sentra.iss). CreateProcess
    # — which is what Popen uses — refuses that combination outright with
    # ERROR_ELEVATION_REQUIRED and never shows a UAC prompt; only ShellExecute
    # knows how to ask the user to elevate.
    #
    # /SILENT keeps the wizard out of the way but still shows a progress
    # window, so the update never looks like a frozen application.
    try:
        import ctypes

        SW_SHOWNORMAL = 1
        result = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", str(path), "/SILENT /NORESTART", None, SW_SHOWNORMAL
        )
    except Exception as exc:  # noqa: BLE001
        return {**state(), "error": f"Could not start the installer: {exc}"}

    # ShellExecuteW returns a value <= 32 to mean failure; the number says why.
    if result <= 32:
        if result == 5:  # SE_ERR_ACCESSDENIED — the UAC prompt was declined
            message = (
                "Installing the update needs administrator permission. "
                "Choose Yes on the Windows prompt and try again."
            )
        else:
            message = f"Could not start the installer (Windows error {result})."
        return {**state(), "error": message}

    _set(status="installing")

    # Step out of the installer's way. It replaces this executable and stops
    # the running copy itself, but doing it here means the handover never
    # depends on that timing — and it lets this response reach the browser
    # first, so the dashboard shows "installing" rather than dying mid-request.
    def _quit_for_installer() -> None:
        time.sleep(3)
        os._exit(0)

    threading.Thread(target=_quit_for_installer, name="sentra-quit", daemon=True).start()
    return state()


def _running_app_bundle() -> Path | None:
    """The Sentra.app directory this process is running from, or None.

    A frozen macOS build's sys.executable is
    ``<...>/Sentra.app/Contents/MacOS/Sentra`` — three parents up is the
    bundle itself. Resolved from the running executable rather than assumed
    to be ``/Applications/Sentra.app`` so this still works if someone put it
    somewhere else, the same self-location trick real macOS self-updaters use.
    None from source (nothing to replace) or on any non-frozen run.
    """
    if sys.platform != "darwin" or not getattr(sys, "frozen", False):
        return None
    bundle = Path(sys.executable).resolve().parents[2]
    return bundle if bundle.suffix == ".app" and bundle.is_dir() else None


def _hdiutil_attach(dmg_path: Path) -> Path:
    """Mount a disk image read-only and return its mount point.

    Parsed from hdiutil's own plist output rather than assumed — hdiutil picks
    the mount point itself (``/Volumes/<name>``, suffixed if that name is
    already taken, e.g. the user still has an earlier download mounted), so
    guessing it would be fragile.
    """
    result = subprocess.run(
        ["hdiutil", "attach", str(dmg_path), "-nobrowse", "-readonly", "-plist"],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"hdiutil attach failed: {result.stderr.decode(errors='replace').strip()}"
        )
    plist = plistlib.loads(result.stdout)
    for entity in plist.get("system-entities", []):
        mount_point = entity.get("mount-point")
        if mount_point:
            return Path(mount_point)
    raise RuntimeError("hdiutil attach reported no mount point")


def _hdiutil_detach(mount_point: Path) -> None:
    # Best-effort: a failed detach leaves a mounted volume behind, which is
    # untidy but not unsafe, and must never be the reason an update that
    # otherwise succeeded gets reported as failed.
    try:
        subprocess.run(
            ["hdiutil", "detach", str(mount_point), "-quiet"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _install_macos(dmg_path: Path) -> dict:
    """Swap the running Sentra.app for the one inside the downloaded .dmg.

    There is no separate installer program on macOS the way there is on
    Windows — the disk image just contains the new Sentra.app, the same shape
    macos/build_macos.sh produces for a manual drag-install. This mounts it
    read-only, copies the new app bundle over the running one with ``ditto``
    (not shutil.copytree — ditto is what preserves the extended attributes and
    resource forks a code-signed bundle needs; a naive copy can leave a bundle
    Gatekeeper refuses to launch), swaps it into place by rename, and relaunches.

    Runs unelevated, matching how the app got there in the first place — the
    signed-in user dragged it into /Applications without sudo. If that same
    user cannot write there now (a different admin installed it, or it was
    moved somewhere with tighter permissions), this fails with an actionable
    message rather than silently doing nothing — the macOS equivalent of the
    Windows "UAC prompt declined" branch above, not a lesser-effort fallback.
    """
    app_path = _running_app_bundle()
    if app_path is None:
        return {
            **state(),
            "error": "Could not determine where Sentra is installed. Open the "
            f"downloaded disk image and drag Sentra to Applications yourself: {dmg_path}",
        }

    mount_point: Path | None = None
    staged_new: Path | None = None
    try:
        mount_point = _hdiutil_attach(dmg_path)
        source_app = mount_point / "Sentra.app"
        if not source_app.is_dir():
            raise RuntimeError(f"Sentra.app not found inside the disk image at {mount_point}")

        # Copy to a sibling temp name first, then swap by rename. The rename
        # is what keeps the real location from ever holding a half-copied
        # bundle, and keeps the old one around to restore if anything after
        # the copy fails.
        staged_new = app_path.parent / f"{app_path.name}.update-{os.getpid()}"
        shutil.rmtree(staged_new, ignore_errors=True)
        result = subprocess.run(
            ["ditto", str(source_app), str(staged_new)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ditto failed: {result.stderr.strip() or result.returncode}")

        old_app = app_path.parent / f"{app_path.name}.previous-{os.getpid()}"
        app_path.rename(old_app)
        try:
            staged_new.rename(app_path)
        except OSError:
            old_app.rename(app_path)  # restore the running copy before giving up
            raise
        shutil.rmtree(old_app, ignore_errors=True)
        staged_new = None  # renamed into place — nothing left for the finally block

    except PermissionError:
        return {
            **state(),
            "error": (
                f"Sentra could not replace itself at {app_path} — this account does not "
                "have permission to write there. Open the downloaded disk image and drag "
                f"Sentra to Applications yourself: {dmg_path}"
            ),
        }
    except Exception as exc:  # noqa: BLE001 — reporting to the operator, not handling
        return {**state(), "error": f"Could not install the update: {exc}"}
    finally:
        if staged_new is not None:
            shutil.rmtree(staged_new, ignore_errors=True)
        if mount_point is not None:
            _hdiutil_detach(mount_point)

    _set(status="installing")

    # Same handover pattern as the Windows branch: step out of the way on a
    # daemon thread so this response reaches the browser before the process
    # exits, rather than the request dying mid-flight.
    def _relaunch_and_quit() -> None:
        time.sleep(1)
        try:
            subprocess.Popen(["open", str(app_path)])
        except OSError:
            pass  # the new bundle is in place regardless; the user can open it by hand
        time.sleep(2)
        os._exit(0)

    threading.Thread(target=_relaunch_and_quit, name="sentra-quit-mac", daemon=True).start()
    return state()


# --- Startup hook -----------------------------------------------------------


def start_background_check(delay: float = STARTUP_CHECK_DELAY_SECONDS) -> None:
    """Schedule the automatic check that runs shortly after launch.

    Fire-and-forget on a daemon thread: an update check must never be something
    the application's startup can block on or fail because of.
    """
    if os.environ.get("SENTRA_DISABLE_UPDATE_CHECK"):
        return

    def _run() -> None:
        time.sleep(delay)
        try:
            check()
        except Exception:  # noqa: BLE001 — a background check must never escalate
            pass

    threading.Thread(target=_run, name="sentra-update-check", daemon=True).start()


def _force_writable(func, path, _exc_info):
    """rmtree error handler: clear the read-only bit and retry once.

    Windows refuses to delete a read-only file and reports it as
    PermissionError. Downloads are not normally read-only, but antivirus and
    Controlled Folder Access both set that bit on quarantined executables —
    which a freshly downloaded installer is a prime candidate to be.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def clear_staged() -> dict:
    """Discard anything downloaded — used when an operator declines an update."""
    try:
        if STAGING_DIR.is_dir():
            shutil.rmtree(STAGING_DIR, onerror=_force_writable)
    except OSError:
        pass
    _set(status="available" if _state.get("latest_version") else "idle",
         staged_path="", downloaded=0, progress=0.0)
    return state()
