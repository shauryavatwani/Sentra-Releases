"""Generate the release manifest the in-app updater reads.

Despite living under windows/, this now covers BOTH platforms: it runs once,
after both the Windows installer and the macOS disk image exist, and writes
one manifest carrying both payloads. It moved out of the windows CI job and
into the publish job for exactly that reason — a manifest built from only one
platform's artifact cannot describe the other one, and this project already
learned once (see CHANGELOG 1.0.2/1.0.3 history) that splitting release
assembly across jobs that do not wait for each other produces a release
that's live before it's actually complete.

Run from the project root after both builds exist:

    python windows/make_release.py <base_url> \\
        --installer windows/output/SentraSetup.exe \\
        --dmg macos/output/Sentra-<version>.dmg \\
        --out windows/output

Writes ``<out>/version.json``, with real sha256 digests for whichever payloads
were given — computed from the files that were actually produced, never typed
in, because a hand-maintained digest eventually drifts and a wrong digest
makes every client refuse the update with an integrity error that looks
exactly like an attack.

The manifest is deliberately provider-neutral: it is a plain JSON file with
https URLs in it. GitHub Releases, S3, Cloudflare R2, Azure Blob or a plain
nginx box all host it identically, so changing provider means changing the
URLs and nothing else.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the built files will be uploaded. Overridden per release by passing a
# base URL as the first argument.
DEFAULT_BASE_URL = "https://github.com/shauryavatwani/Sentra-Releases/releases/latest/download"


def read_version() -> str:
    source = (PROJECT_ROOT / "Formal_Code" / "sentra_version.py").read_text(encoding="utf-8")
    match = re.search(r'^VERSION\s*=\s*"([^"]+)"', source, re.MULTILINE)
    if not match:
        raise SystemExit("Could not read VERSION from Formal_Code/sentra_version.py")
    return match.group(1)


def _changelog_section(version: str) -> str:
    """The CHANGELOG.md section for this version, or "" if there isn't one.

    Sections are ``## <version>`` headings; everything up to the next ``##``
    belongs to that release.
    """
    changelog = PROJECT_ROOT / "CHANGELOG.md"
    try:
        lines = changelog.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    collected: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip().lstrip("vV").startswith(version)
            continue
        if inside:
            collected.append(line)

    return "\n".join(collected).strip()


def _first_paragraph(text: str) -> str:
    """One-line summary for the manifest's `notes`, shown in-app before installing."""
    for block in text.split("\n\n"):
        cleaned = " ".join(
            line.lstrip("-*# ").strip() for line in block.strip().splitlines()
        ).strip()
        if cleaned:
            return cleaned[:300]
    return ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_dmg(pattern_or_path: str | None, version: str) -> Path | None:
    """Accept an exact path or a glob (CI passes a glob since the exact
    version-stamped filename isn't known until this script reads it)."""
    if not pattern_or_path:
        default = PROJECT_ROOT / "macos" / "output" / f"Sentra-{version}.dmg"
        return default if default.is_file() else None
    direct = Path(pattern_or_path)
    if direct.is_file():
        return direct
    matches = sorted(glob.glob(pattern_or_path))
    return Path(matches[0]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--installer",
        default=str(PROJECT_ROOT / "windows" / "output" / "SentraSetup.exe"),
        help="path to the built Windows installer",
    )
    parser.add_argument(
        "--dmg",
        default=None,
        help="path or glob to the built macOS disk image "
        "(default: macos/output/Sentra-<version>.dmg if present)",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "windows" / "output"),
        help="directory to write version.json / release_notes.md into",
    )
    args = parser.parse_args()

    installer = Path(args.installer)
    if not installer.is_file():
        raise SystemExit(
            f"Windows installer not found at {installer}.\n"
            "The Windows build must exist before a manifest can be written — "
            "a manifest without a real digest is worse than no manifest."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    version = read_version()
    build_date = __import__("datetime").date.today().isoformat()

    # Two copies of the Windows installer's bytes, published under two names:
    #
    #   SentraSetup-<version>.exe  is what the manifest points at. Version in
    #       the name so several releases coexist in one bucket and a client
    #       that downloaded 1.2.0 can never be handed a 1.3.0 body under the
    #       old name by a cache — which would fail the checksum and look like
    #       tampering.
    #   SentraSetup.exe            is the human link. GitHub serves
    #       /releases/latest/download/<name>, so an unversioned name gives a
    #       permanent "download the current version" URL a README can point
    #       at without being edited on every release.
    windows_release_name = f"SentraSetup-{version}.exe"
    windows_stamped = out_dir / windows_release_name
    if windows_stamped.resolve() != installer.resolve():
        windows_stamped.write_bytes(installer.read_bytes())

    windows_entry = {
        "url": f"{base_url}/{windows_release_name}",
        "sha256": sha256_of(windows_stamped),
        "size": windows_stamped.stat().st_size,
    }

    platforms = {"win32": dict(windows_entry)}

    # The macOS disk image is already named Sentra-<version>.dmg by
    # macos/build_macos.sh — no extra versioned copy needed the way Windows's
    # unversioned convenience name required one.
    dmg = _resolve_dmg(args.dmg, version)
    if dmg is not None:
        dmg_release_name = dmg.name
        dmg_staged = out_dir / dmg_release_name
        if dmg_staged.resolve() != dmg.resolve():
            dmg_staged.write_bytes(dmg.read_bytes())
        platforms["darwin"] = {
            "url": f"{base_url}/{dmg_release_name}",
            "sha256": sha256_of(dmg_staged),
            "size": dmg_staged.stat().st_size,
        }
    else:
        print(
            "WARNING: no macOS disk image given/found — publishing a Windows-only "
            "manifest. Mac clients running the new (post-1.0.3) updater will see "
            "'No update is available for this platform yet.' until a mac build "
            "is included."
        )

    # Flat top-level fields stay the Windows payload — that is what they meant
    # before "platforms" existed, and an already-installed 1.0.2/1.0.3 client's
    # updater only ever reads these, with no idea "platforms" exists.
    manifest = {
        "version": version,
        **windows_entry,
        "build_date": build_date,
        "notes": "",
        "notes_url": f"{base_url}/release_notes.md",
        "mandatory": False,
        "platforms": platforms,
    }

    manifest_path = out_dir / "version.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Release notes come from CHANGELOG.md when it has a section for this
    # version, so what shows on the GitHub release page is written
    # deliberately alongside the code rather than typed into a web form after
    # the fact. Falls back to a stub only when the changelog has nothing to
    # say about this version.
    notes_path = out_dir / "release_notes.md"
    notes_body = _changelog_section(version)
    if notes_body:
        notes_path.write_text(notes_body, encoding="utf-8")
        manifest["notes"] = _first_paragraph(notes_body)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif not notes_path.exists():
        notes_path.write_text(
            f"# Sentra {version}\n\nReleased {build_date}.\n", encoding="utf-8"
        )

    print(f"  version    : {version}")
    print(f"  windows    : {windows_release_name} ({windows_entry['size'] / (1024*1024):.0f} MB)")
    if "darwin" in platforms:
        print(
            f"  macos      : {platforms['darwin']['url'].rsplit('/', 1)[-1]} "
            f"({platforms['darwin']['size'] / (1024*1024):.0f} MB)"
        )
    print(f"  manifest   : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
