"""Update checking and staging for Sentra.

Deliberately built around one idea: **an update feed is just a JSON file at a
URL.** Nothing here knows what a GitHub Release, an S3 bucket or a Cloudflare R2
object is, so moving between them is a URL change in
``Database/update_config.json`` and never a code change.

Manifest schema (host this as ``version.json`` next to the installer)::

    {
      "version":    "1.1.0",              # required
      "url":        "https://.../SentraSetup-1.1.0.exe",   # required
      "sha256":     "9f86d081884c7d...",  # required — see below
      "size":       734003200,            # optional, for the progress bar
      "build_date": "2026-08-15",         # optional, shown in the UI
      "notes":      "Fixed the camera reconnect loop.",     # optional
      "notes_url":  "https://.../release_notes.md",         # optional
      "mandatory":  false                 # optional, styles the prompt louder
    }

``sha256`` is **required and enforced**. This module downloads an executable and
then runs it with administrator rights, so an unverified download would hand
anyone who can spoof the feed host a way to run code on every client machine.
A manifest without a digest is rejected rather than trusted, and HTTP feeds are
refused outright — the check is only as trustworthy as its transport.

Nothing is ever installed automatically. The check runs by itself, the download
runs when the operator asks, and applying it is always an explicit click.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    "supported": sys.platform == "win32",
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


def _validate(manifest: dict) -> tuple[str, str, str]:
    """Return (version, url, sha256) or raise ValueError with a usable message."""
    version = str(manifest.get("version", "")).strip()
    url = str(manifest.get("url", "")).strip()
    digest = str(manifest.get("sha256", "")).strip().lower()

    if not version:
        raise ValueError("The update manifest is missing a version number.")
    if not url.lower().startswith("https://"):
        raise ValueError("The update manifest's download URL must be https://.")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(
            "The update manifest has no valid sha256 checksum. "
            "Sentra will not install an update it cannot verify."
        )
    return version, url, digest


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
        version, url, digest = _validate(manifest)
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
        size=int(manifest.get("size") or 0),
        last_checked=time.time(),
        error="",
    )

    if not newer:
        _set(status="up_to_date", staged_path="", downloaded=0, progress=0.0)
        return state()

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
    candidate = STAGING_DIR / f"SentraSetup-{version}.exe"
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
    target = STAGING_DIR / f"SentraSetup-{version}.exe"
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
    """Staged installers are ~1GB each; leaving every past one is not polite."""
    try:
        for leftover in STAGING_DIR.glob("SentraSetup-*.exe"):
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
    """Hand the staged installer control and step out of its way.

    The installer stops the running Sentra itself (``PrepareToInstall`` in
    ``windows/sentra.iss``), replaces the program files, and leaves every folder
    under ``ProgramData\\Sentra`` — the database, registered faces, visitor
    photos, camera configuration — exactly as it found them.
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

    if sys.platform != "win32":
        return {
            **state(),
            "error": "Automatic installation is only supported on Windows. "
            f"Run the downloaded installer manually: {path}",
        }

    try:
        # /SILENT keeps the wizard out of the way but still shows a progress
        # window, so the update never looks like a frozen application.
        subprocess.Popen(
            [str(path), "/SILENT", "/NORESTART"],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except OSError as exc:
        return {**state(), "error": f"Could not start the installer: {exc}"}

    _set(status="installing")
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
