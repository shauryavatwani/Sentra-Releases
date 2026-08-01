"""Backwards-compatible single-camera view over the multi-camera store.

Sentra used to support exactly one camera and this module owned its config.
Storage now lives in `camera_store.py` (a list of cameras); this module stays
as a thin facade over the *first* camera so older callers keep working
unchanged.

Prefer `camera_store` in new code — this only ever sees one camera and cannot
express the others.
"""

from __future__ import annotations

import camera_store
from camera_store import DEFAULT_NAME, DEFAULT_RTSP_URL  # re-exported

__all__ = ["load_camera_config", "save_camera_config", "DEFAULT_NAME", "DEFAULT_RTSP_URL"]


def load_camera_config() -> dict:
    """The first configured camera, as the old flat {name, rtsp_url} dict."""
    cameras = camera_store.load_cameras()  # always returns at least one
    first = cameras[0]
    return {"name": first["name"], "rtsp_url": first["rtsp_url"]}


def save_camera_config(name: str, rtsp_url: str) -> dict:
    """Rename / re-point the first camera, leaving any others untouched."""
    cameras = camera_store.load_cameras()
    updated = camera_store.update_camera(cameras[0]["id"], name=name, rtsp_url=rtsp_url)
    return {"name": updated["name"], "rtsp_url": updated["rtsp_url"]}
