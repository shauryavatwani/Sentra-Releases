"""Ask a camera, over ONVIF, what its RTSP URLs actually are.

This is the logic that used to live in the top-level ``get_rtsp_url.py``
script, turned into something importable. That script hardcoded one IP and
password and only printed — it could not be reused by the dashboard, which is
what the Cameras page needs in order to discover a URL for the user.

``get_rtsp_url.py`` still works as a command-line tool and now calls in here.

Why this exists at all: the RTSP path is vendor-specific (this CP Plus camera
uses ``/live/channel0`` and ``/live/channel1``) and the IP moves whenever DHCP
reassigns it. Asking the camera beats making the user find a vendor tool.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

DEFAULT_ONVIF_PORT = 8000
DEFAULT_USERNAME = "admin"


class DiscoveryError(Exception):
    """Discovery failed, with a message safe to show a non-technical user."""


@dataclass
class StreamProfile:
    name: str
    width: int
    height: int
    url: str

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["resolution"] = self.resolution
        return data


def _friendly_error(exc: Exception, ip: str) -> str:
    """Turn a zeep/ONVIF stack trace into something actionable.

    These failures are the normal case when someone mistypes a password or the
    camera has moved, so they must not surface as a raw exception in the UI.
    """
    text = str(exc).lower()

    if "not authorized" in text or "unauthorized" in text or "auth" in text:
        return (
            "The camera rejected those credentials. Check the username and "
            "password. Note that some cameras temporarily lock out ONVIF "
            "after several failed attempts — if you are sure they are right, "
            "power-cycle the camera and try again."
        )
    if any(s in text for s in ("timed out", "timeout", "unreachable",
                               "no route", "connection refused", "connect")):
        return (
            f"Could not reach {ip}. Check the camera is powered on, that this "
            "machine is on the same network, and that the IP is current — it "
            "changes when the router reassigns addresses."
        )
    if "name or service not known" in text or "nodename" in text:
        return f"{ip} is not a valid address."
    return f"Could not query the camera over ONVIF: {exc}"


def discover_rtsp_urls(
    ip: str,
    username: str = DEFAULT_USERNAME,
    password: str = "",
    port: int = DEFAULT_ONVIF_PORT,
) -> tuple[list[StreamProfile], str]:
    """Return ``(profiles, device_description)`` for the camera at *ip*.

    Profiles come back in the camera's own order, which for this hardware is
    main stream first, sub-stream second.

    Raises DiscoveryError with a user-facing message on any failure.
    """
    ip = (ip or "").strip()
    if not ip:
        raise DiscoveryError("Enter the camera's IP address.")

    try:
        from onvif import ONVIFCamera
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise DiscoveryError(
            "ONVIF support is not installed on this machine "
            "(pip install onvif-zeep). Use the manual RTSP option instead."
        ) from exc

    try:
        camera = ONVIFCamera(ip, int(port), username, password)
        info = camera.devicemgmt.GetDeviceInformation()
        device = f"{info.Manufacturer} {info.Model}".strip()
    except Exception as exc:
        raise DiscoveryError(_friendly_error(exc, ip)) from exc

    try:
        media = camera.create_media_service()
        profiles = media.GetProfiles()
    except Exception as exc:
        raise DiscoveryError(_friendly_error(exc, ip)) from exc

    found: list[StreamProfile] = []
    for profile in profiles:
        try:
            uri = media.GetStreamUri(
                {
                    "StreamSetup": {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"},
                    },
                    "ProfileToken": profile.token,
                }
            )
        except Exception:
            # One unreadable profile shouldn't lose the others — a camera
            # advertising a stream it won't serve is common enough.
            continue

        encoder = getattr(profile, "VideoEncoderConfiguration", None)
        resolution = getattr(encoder, "Resolution", None) if encoder else None
        found.append(
            StreamProfile(
                name=str(getattr(profile, "Name", "") or profile.token),
                width=int(getattr(resolution, "Width", 0) or 0),
                height=int(getattr(resolution, "Height", 0) or 0),
                url=str(uri.Uri),
            )
        )

    if not found:
        raise DiscoveryError(
            "Connected to the camera, but it reported no RTSP streams. "
            "Enter the URL manually instead."
        )
    return found, device


def best_profile(profiles: list[StreamProfile]) -> StreamProfile:
    """Pick the profile to use by default.

    Deliberately the *smallest* stream, not the largest: decoding the full
    2304x1296 main stream continuously cost ~150-165% CPU versus ~69% for the
    640x360 sub-stream, for no detection benefit at the size frames are
    resized to anyway. The user can still pick another profile in the UI.
    """
    with_size = [p for p in profiles if p.width and p.height]
    if not with_size:
        return profiles[0]
    return min(with_size, key=lambda p: p.width * p.height)
