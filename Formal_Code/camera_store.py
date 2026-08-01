"""Storage for every configured camera.

Supersedes the original single-camera `camera_config.py`, which stored one
flat ``{name, rtsp_url}`` object. That module still exists and still works —
it now reads through this one — so nothing that imported it had to change.

File format (``Database/camera_config.json``)::

    {
      "version": 2,
      "cameras": [
        {
          "id": "cam_ab12cd34",
          "name": "Main Gate",
          "location": "Building A entrance",
          "description": "",
          "rtsp_url": "rtsp://192.168.0.113:5543/live/channel1",
          "ai_enabled": true,
          "connection_method": "onvif",
          "onvif": {"ip": "...", "port": 8000,
                    "username": "admin", "password": "..."}
        }
      ]
    }

A v1 file is migrated into a one-element v2 list on first read, so an existing
install keeps its camera, its name and its URL without the user doing anything.

**Credentials**: ONVIF username/password are stored so the RTSP URL can be
re-discovered when the camera's IP changes (DHCP moves it more often than you
would like). They are therefore written in plaintext, and the file is chmod
0600 on POSIX to limit that. `public_camera()` strips the password from
anything the API returns — it exposes ``has_password`` instead, so the UI can
show "credentials saved" without the value ever reaching a browser.
"""

from __future__ import annotations

import json
import os
import secrets
import threading

from sentra_paths import CAMERA_CONFIG_FILE as CONFIG_FILE

DEFAULT_NAME = "Shark Tank Pitch Camera"
DEFAULT_RTSP_URL = "rtsp://192.168.1.9:5543/live/channel0"
DEFAULT_ONVIF_PORT = 8000

CONNECTION_METHODS = ("onvif", "manual")

# Writes are read-modify-write on a single file, and both the API (multiple
# request threads) and the engine can touch it.
_lock = threading.RLock()


def _new_id() -> str:
    return f"cam_{secrets.token_hex(4)}"


def _blank_onvif() -> dict:
    return {"ip": "", "port": DEFAULT_ONVIF_PORT, "username": "", "password": ""}


def _normalise(raw: dict) -> dict:
    """Coerce one stored entry into the full shape, filling in missing keys.

    Written defensively because this file is user-editable and older versions
    of it genuinely lack most of these fields.
    """
    onvif_raw = raw.get("onvif") or {}
    method = str(raw.get("connection_method") or "manual").lower()
    if method not in CONNECTION_METHODS:
        method = "manual"

    try:
        port = int(onvif_raw.get("port") or DEFAULT_ONVIF_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_ONVIF_PORT

    return {
        "id": str(raw.get("id") or _new_id()),
        "name": str(raw.get("name") or DEFAULT_NAME).strip() or DEFAULT_NAME,
        "location": str(raw.get("location") or "").strip(),
        "description": str(raw.get("description") or "").strip(),
        "rtsp_url": str(raw.get("rtsp_url") or "").strip(),
        # Default True so a migrated v1 camera keeps being analysed exactly as
        # it was before this module existed.
        "ai_enabled": bool(raw.get("ai_enabled", True)),
        "connection_method": method,
        "onvif": {
            "ip": str(onvif_raw.get("ip") or "").strip(),
            "port": port,
            "username": str(onvif_raw.get("username") or "").strip(),
            "password": str(onvif_raw.get("password") or ""),
        },
    }


def _read_raw() -> dict:
    if not CONFIG_FILE.is_file():
        return {"version": 2, "cameras": []}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 2, "cameras": []}

    if isinstance(data, dict) and isinstance(data.get("cameras"), list):
        return {"version": 2, "cameras": data["cameras"]}

    # --- v1 migration -----------------------------------------------------
    # The old format was a flat {"name", "rtsp_url"} object. Preserve it as the
    # first (and initially only) camera rather than discarding the user's URL.
    if isinstance(data, dict) and ("rtsp_url" in data or "name" in data):
        return {
            "version": 2,
            "cameras": [
                {
                    "id": _new_id(),
                    "name": data.get("name") or DEFAULT_NAME,
                    "rtsp_url": data.get("rtsp_url") or DEFAULT_RTSP_URL,
                    "ai_enabled": True,
                    "connection_method": "manual",
                }
            ],
        }

    return {"version": 2, "cameras": []}


def _write(cameras: list[dict]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 2, "cameras": cameras}
    CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Camera passwords live in here.
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass  # best effort; Windows ACLs don't map onto this


def load_cameras() -> list[dict]:
    """Every configured camera, migrating and back-filling as needed.

    Guarantees at least one camera exists, so the dashboard and engine always
    have something to show rather than a confusing empty state on a fresh
    install.
    """
    with _lock:
        raw = _read_raw()
        cameras = [_normalise(entry) for entry in raw["cameras"]]

        if not cameras:
            cameras = [
                _normalise(
                    {
                        "name": DEFAULT_NAME,
                        "rtsp_url": DEFAULT_RTSP_URL,
                        "connection_method": "manual",
                    }
                )
            ]

        # Persist the migrated/defaulted shape so the next read is a plain load
        # and the file on disk matches what callers just received.
        if raw != {"version": 2, "cameras": cameras}:
            _write(cameras)
        return cameras


def get_camera(camera_id: str) -> dict | None:
    return next((c for c in load_cameras() if c["id"] == camera_id), None)


def ai_cameras() -> list[dict]:
    """Cameras the engine should run face/fight detection on."""
    return [c for c in load_cameras() if c["ai_enabled"] and c["rtsp_url"]]


def _validate(name: str, rtsp_url: str, method: str) -> tuple[str, str, str]:
    name = (name or "").strip()
    rtsp_url = (rtsp_url or "").strip()
    method = (method or "manual").lower()

    if not name:
        raise ValueError("Camera name cannot be empty.")
    if method not in CONNECTION_METHODS:
        raise ValueError(f"Unknown connection method: {method!r}")
    if not rtsp_url:
        # True for both methods: ONVIF discovery must have produced a URL
        # before the camera is saved, otherwise there is nothing to connect to.
        raise ValueError("An RTSP URL is required. Run discovery, or enter one manually.")
    if not rtsp_url.lower().startswith("rtsp://"):
        raise ValueError("RTSP URL must start with rtsp://")
    return name, rtsp_url, method


def add_camera(
    name: str,
    rtsp_url: str,
    *,
    location: str = "",
    description: str = "",
    connection_method: str = "manual",
    onvif: dict | None = None,
    ai_enabled: bool = True,
) -> dict:
    name, rtsp_url, connection_method = _validate(name, rtsp_url, connection_method)
    with _lock:
        cameras = load_cameras()
        camera = _normalise(
            {
                "id": _new_id(),
                "name": name,
                "location": location,
                "description": description,
                "rtsp_url": rtsp_url,
                "ai_enabled": ai_enabled,
                "connection_method": connection_method,
                "onvif": onvif or _blank_onvif(),
            }
        )
        cameras.append(camera)
        _write(cameras)
        return camera


def update_camera(camera_id: str, **fields) -> dict:
    """Patch one camera. Only supplied fields change.

    A blank ``onvif.password`` is treated as "leave the stored one alone", so
    the UI can render the form without ever holding the real password.
    """
    with _lock:
        cameras = load_cameras()
        index = next((i for i, c in enumerate(cameras) if c["id"] == camera_id), None)
        if index is None:
            raise KeyError(camera_id)

        current = cameras[index]
        merged = dict(current)

        for key in ("name", "location", "description", "rtsp_url",
                    "connection_method", "ai_enabled"):
            if key in fields and fields[key] is not None:
                merged[key] = fields[key]

        if fields.get("onvif") is not None:
            incoming = dict(fields["onvif"])
            if not incoming.get("password"):
                incoming["password"] = current["onvif"]["password"]
            merged["onvif"] = incoming

        merged["name"], merged["rtsp_url"], merged["connection_method"] = _validate(
            merged["name"], merged["rtsp_url"], merged["connection_method"]
        )

        cameras[index] = _normalise(merged)
        _write(cameras)
        return cameras[index]


def delete_camera(camera_id: str) -> bool:
    """Remove a camera. Refuses to delete the last one.

    An empty camera list would leave the dashboard with nothing to show and the
    engine with nothing to open, which reads as a broken app rather than a
    deliberate state.
    """
    with _lock:
        cameras = load_cameras()
        if len(cameras) <= 1:
            raise ValueError("Cannot remove the only camera — add another one first.")
        remaining = [c for c in cameras if c["id"] != camera_id]
        if len(remaining) == len(cameras):
            return False
        _write(remaining)
        return True


def public_camera(camera: dict) -> dict:
    """The API-safe view: same data, password replaced by a boolean."""
    onvif = camera.get("onvif") or {}
    return {
        "id": camera["id"],
        "name": camera["name"],
        "location": camera.get("location", ""),
        "description": camera.get("description", ""),
        "rtsp_url": camera["rtsp_url"],
        "ai_enabled": camera.get("ai_enabled", True),
        "connection_method": camera.get("connection_method", "manual"),
        "onvif": {
            "ip": onvif.get("ip", ""),
            "port": onvif.get("port", DEFAULT_ONVIF_PORT),
            "username": onvif.get("username", ""),
            "has_password": bool(onvif.get("password")),
        },
    }
