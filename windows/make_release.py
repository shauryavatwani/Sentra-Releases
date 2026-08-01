"""Generate the release manifest the in-app updater reads.

Run from the project root after the installer has been built:

    python windows/make_release.py

Writes ``windows/output/version.json`` next to ``SentraSetup.exe``, containing
the installer's real sha256. The checksum is computed from the file that was
actually produced, never typed in — a hand-maintained digest eventually drifts,
and a wrong digest makes every client refuse the update with an integrity error
that looks exactly like an attack.

The manifest is deliberately provider-neutral: it is a plain JSON file with an
https URL in it. GitHub Releases, S3, Cloudflare R2, Azure Blob or a plain nginx
box all host it identically, so changing provider means changing the URL and
nothing else.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "windows" / "output"
INSTALLER = OUTPUT_DIR / "SentraSetup.exe"

# Where the built files will be uploaded. Overridden per release by editing
# this constant or by passing a base URL as the first argument.
DEFAULT_BASE_URL = "https://github.com/shauryavatwani/Sentra-Releases/releases/latest/download"


def read_version() -> str:
    source = (PROJECT_ROOT / "Formal_Code" / "sentra_version.py").read_text(encoding="utf-8")
    match = re.search(r'^VERSION\s*=\s*"([^"]+)"', source, re.MULTILINE)
    if not match:
        raise SystemExit("Could not read VERSION from Formal_Code/sentra_version.py")
    return match.group(1)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not INSTALLER.is_file():
        raise SystemExit(
            f"Installer not found at {INSTALLER}.\n"
            "Run windows\\build_windows.bat first — this script describes the "
            "installer it produces."
        )

    base_url = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL).rstrip("/")
    version = read_version()
    build_date = __import__("datetime").date.today().isoformat()

    # Version-stamped filename so several releases can sit in one bucket and a
    # client that downloaded 1.2.0 never receives a 1.3.0 body under the old
    # name from a cache.
    release_name = f"SentraSetup-{version}.exe"
    stamped = OUTPUT_DIR / release_name
    if stamped.resolve() != INSTALLER.resolve():
        stamped.write_bytes(INSTALLER.read_bytes())

    manifest = {
        "version": version,
        "url": f"{base_url}/{release_name}",
        "sha256": sha256_of(stamped),
        "size": stamped.stat().st_size,
        "build_date": build_date,
        "notes": "",
        "notes_url": f"{base_url}/release_notes.md",
        "mandatory": False,
    }

    manifest_path = OUTPUT_DIR / "version.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    notes_path = OUTPUT_DIR / "release_notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            f"# Sentra {version}\n\nReleased {build_date}.\n\n- \n",
            encoding="utf-8",
        )

    (OUTPUT_DIR / "PUBLISHING.txt").write_text(
        f"""Publishing Sentra {version}
{'=' * (18 + len(version))}

Upload these three files to the SAME folder/bucket/release:

    {release_name}
    version.json
    release_notes.md

Then confirm the URL inside version.json actually resolves:

    {manifest['url']}

The client's copy of Sentra reads version.json on startup, compares the
version against its own, and offers the update. It refuses any download
whose sha256 does not match the manifest, so re-upload both files together
whenever the installer changes — a fresh installer with a stale manifest
will be rejected by every client.

Changing host later (S3, R2, your own server) needs no new build. Point
Database/update_config.json on the client at the new manifest:

    {{ "feed_url": "https://your-new-host/version.json" }}

or set the SENTRA_UPDATE_URL environment variable.

Before uploading, write the actual changes into release_notes.md and copy
the summary line into the "notes" field of version.json — that text is what
the operator reads in Settings before deciding to install.
""",
        encoding="utf-8",
    )

    print(f"  version    : {version}")
    print(f"  installer  : {release_name} ({manifest['size'] / (1024*1024):.0f} MB)")
    print(f"  sha256     : {manifest['sha256']}")
    print(f"  manifest   : {manifest_path}")
    print(f"  next steps : {OUTPUT_DIR / 'PUBLISHING.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
