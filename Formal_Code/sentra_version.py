"""Version identity for Sentra — the one place a version number is written.

Everything downstream reads from here rather than repeating the literal:

* ``backend_v2/main.py`` serves it at ``/api/version`` for the About panel.
* ``Formal_Code/updater.py`` compares it against the published manifest.
* ``windows/sentra.spec`` writes ``windows/version_define.iss`` from it at build
  time, which ``windows/sentra.iss`` includes — so the installer, the About
  panel and the update check can never disagree about what version this is.

Bump :data:`VERSION` here and every one of those follows.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import sentra_paths

# --- Identity ---------------------------------------------------------------

APP_NAME = "Sentra"
VERSION = "1.0.1"
PUBLISHER = "Delhi Public School Bangalore East"

# Where the update manifest lives by default. Deliberately just a URL to a JSON
# file: GitHub Releases, S3, R2, Azure Blob and a plain nginx box all serve one
# identically, so switching provider is a config change and never a code change.
# See updater.py for the manifest schema.
DEFAULT_UPDATE_FEED = (
    "https://github.com/shauryavatwani/Sentra-Releases/releases/latest/download/version.json"
)

# Overridable without a rebuild, in priority order:
#   1. SENTRA_UPDATE_URL environment variable
#   2. <DATA_ROOT>/Database/update_config.json  -> {"feed_url": "..."}
#   3. DEFAULT_UPDATE_FEED above
UPDATE_CONFIG_FILE = sentra_paths.DATABASE_DIR / "update_config.json"


def _build_info() -> dict:
    """Build metadata stamped by the PyInstaller build, if this is one.

    A source checkout has no stamp, so the date falls back to this file's own
    mtime — honest ("when this code was last touched") rather than inventing a
    build date that never happened.
    """
    stamp = sentra_paths.RESOURCE_ROOT / "build_info.json"
    try:
        return json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_INFO = _build_info()

BUILD_DATE: str = _INFO.get("build_date") or _dt.date.fromtimestamp(
    Path(__file__).stat().st_mtime
).isoformat()

# "release" for a packaged build, "source" when run from a checkout. The About
# panel shows this so a developer never mistakes their working tree for the
# build the client is actually running.
BUILD_CHANNEL: str = _INFO.get("channel") or ("release" if _INFO else "source")


def update_feed_url() -> str:
    """Resolve the update manifest URL, honouring the override chain."""
    env = os.environ.get("SENTRA_UPDATE_URL")
    if env:
        return env.strip()

    try:
        config = json.loads(UPDATE_CONFIG_FILE.read_text(encoding="utf-8"))
        url = config.get("feed_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    except (OSError, ValueError):
        pass

    return DEFAULT_UPDATE_FEED


# --- Comparison -------------------------------------------------------------


def parse(version: str) -> tuple[int, ...]:
    """Turn ``"1.2.3"`` into ``(1, 2, 3)``.

    Non-numeric trailing parts (``"1.2.0-beta"``) are truncated at the first
    piece that is not a plain integer, so a malformed manifest degrades to a
    coarser comparison instead of raising during a background check.
    """
    parts: list[int] = []
    for chunk in str(version).strip().lstrip("vV").split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def is_newer(candidate: str, current: str = VERSION) -> bool:
    """True when ``candidate`` is a strictly higher version than ``current``.

    Pads the shorter tuple so ``1.1`` and ``1.1.0`` compare equal rather than
    the shorter one reading as older and offering a pointless update.
    """
    a, b = parse(candidate), parse(current)
    width = max(len(a), len(b))
    a += (0,) * (width - len(a))
    b += (0,) * (width - len(b))
    return a > b


def describe() -> dict:
    """The payload behind ``/api/version`` and the About panel."""
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "build_date": BUILD_DATE,
        "channel": BUILD_CHANNEL,
        "publisher": PUBLISHER,
        "platform": _INFO.get("platform", ""),
    }
