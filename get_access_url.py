"""Find the URL other devices on the same wifi should use to reach Sentra.

The Mac's LAN IP changes every time it joins a different wifi network (DHCP
reassigns it), so a hardcoded URL like http://Shauryas-MacBook-Air.local:8000
only works reliably from other Apple devices (Android/Windows often can't
resolve .local mDNS names at all). This script finds the actual current IP
instead, which works from any device.

Run any time you switch wifi networks, right before showing someone the app:

    python3 get_access_url.py [--port 8000]
"""

import argparse
import socket
import sys
import urllib.error
import urllib.request


def get_lan_ip() -> str:
    # Doesn't actually send packets (UDP "connect" just picks a route) — this
    # is the standard trick to find the IP of whichever interface the OS
    # would use to reach the outside world, without hardcoding an interface
    # name like en0 (which varies: could be wifi, ethernet, a dongle, etc.)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def check_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        # Any HTTP response (including redirects to /login, 4xx, etc.) means
        # the server answered — that's what we're checking, not auth.
        return True
    except (urllib.error.URLError, socket.timeout, ConnectionError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ip = get_lan_ip()
    url = f"http://{ip}:{args.port}"

    print(f"This Mac's current LAN IP: {ip}")
    print(f"\nShare this with anyone on the same wifi:\n\n    {url}\n")

    if check_reachable(url):
        print("Confirmed: Sentra is up and reachable at that address.")
    else:
        print(
            "WARNING: nothing answered at that address just now — "
            "is Sentra running? (Start Sentra.command)"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
