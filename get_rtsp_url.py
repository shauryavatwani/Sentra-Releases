"""Query a camera's ONVIF service for its current RTSP URLs.

Run this any time a camera's IP changes (e.g. after a router reboot):

    python3 get_rtsp_url.py [IP] [--username admin] [--password sharktank] [--port 8000]

With no arguments it uses the constants below, unchanged from before.

The actual ONVIF logic now lives in Formal_Code/onvif_discovery.py, which the
dashboard's Cameras page also calls for its own "auto-discover" button — this
script is now a thin CLI wrapper over the same code path, not a separate
implementation to keep in sync.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Formal_Code"))
from onvif_discovery import DiscoveryError, discover_rtsp_urls  # noqa: E402

CAMERA_IP = "192.168.1.12"  # <-- used only if no IP is given on the command line
ONVIF_PORT = 8000
USERNAME = "admin"
PASSWORD = "sharktank"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ip", nargs="?", default=CAMERA_IP)
    parser.add_argument("--port", type=int, default=ONVIF_PORT)
    parser.add_argument("--username", default=USERNAME)
    parser.add_argument("--password", default=PASSWORD)
    args = parser.parse_args()

    try:
        profiles, device = discover_rtsp_urls(
            args.ip, args.username, args.password, args.port
        )
    except DiscoveryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Connected: {device}\n")
    for profile in profiles:
        print(f"{profile.name}: {profile.resolution}")
        print(f"  {profile.url}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
