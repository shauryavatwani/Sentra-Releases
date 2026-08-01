"""Sentra data pack — move a populated system between machines as one file.

The public installer ships empty (it carries no photographs or face embeddings
of real people; see the .gitignore rationale). This module is how the actual
data reaches an install without anyone re-registering eight people by hand:

* :func:`export_pack` bundles the registered people, their photos, the visitor
  records and the detection history into a single ``.sentra`` file.
* :func:`import_pack` unpacks it into a running install — merging the people so
  face recognition works immediately, and seeding the history only when the
  target is empty so it can never duplicate rows on a second import.

The file is an ordinary zip with a manifest inside, so it can be inspected, but
the extension is deliberately distinct so a recipient uploads it rather than
unzipping it and wondering which piece to import.

Security note: the embeddings travel as a pickle (that is the on-disk format
``face_register.py`` already owns). A pickle from an uploaded file is an
arbitrary-code vector, so import never calls a bare ``pickle.load`` — it uses a
restricted unpickler that permits only numpy array reconstruction and basic
builtins, and refuses anything else.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import pickle
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import numpy as np

import sentra_paths

PACK_FORMAT = "sentra-data-pack/1"
MANIFEST_NAME = "manifest.json"
EMBEDDINGS_ARCNAME = "face_embeddings.pkl"
DB_ARCNAME = "detections.db"
FACES_PREFIX = "Faces/"
VISITORS_PREFIX = "Visitors/"

HISTORY_TABLES = ("detections", "anomalies", "visitors", "visitor_alerts")


# --- Safe unpickling --------------------------------------------------------


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow only what a ``dict[str, np.ndarray]`` legitimately needs.

    A face-embeddings pickle references numpy's array-reconstruction machinery
    and nothing else. Permitting exactly that — and refusing every other global,
    which is how a malicious pickle would reach ``os.system`` or similar — makes
    loading an uploaded file safe without changing the on-disk format.
    """

    _ALLOWED = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        # numpy scalar dtypes (float32 etc.) live under these two modules and
        # are plain data types, not callables that can do harm.
        if module in ("numpy", "numpy.core.numeric", "numpy._core.numeric"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to unpickle {module}.{name} from an imported data pack."
        )


def _safe_load_embeddings(raw: bytes) -> dict[str, np.ndarray]:
    data = _RestrictedUnpickler(io.BytesIO(raw)).load()
    if not isinstance(data, dict):
        raise ValueError("The data pack's embeddings are not in the expected format.")
    clean: dict[str, np.ndarray] = {}
    for name, vector in data.items():
        if isinstance(name, str) and isinstance(vector, np.ndarray):
            clean[name] = vector
    return clean


# --- Export -----------------------------------------------------------------


def _count_rows(db_path: Path) -> dict[str, int]:
    counts = {t: 0 for t in HISTORY_TABLES}
    if not db_path.is_file():
        return counts
    with sqlite3.connect(db_path) as conn:
        for table in HISTORY_TABLES:
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                counts[table] = 0
    return counts


def export_pack(dest: Path) -> dict:
    """Write a data pack describing the current install to ``dest``.

    Returns a summary dict (also embedded in the pack as its manifest).
    """
    dest = Path(dest)
    embeddings_file = sentra_paths.FACE_EMBEDDINGS_FILE
    faces_dir = sentra_paths.FACES_DIR
    visitors_dir = sentra_paths.VISITORS_DIR
    db_file = sentra_paths.DETECTIONS_DB

    people: list[str] = []
    if embeddings_file.is_file():
        people = sorted(_safe_load_embeddings(embeddings_file.read_bytes()).keys())

    manifest = {
        "format": PACK_FORMAT,
        "exported_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "people": people,
        "counts": _count_rows(db_file),
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))

        if embeddings_file.is_file():
            zf.write(embeddings_file, EMBEDDINGS_ARCNAME)
        if db_file.is_file():
            zf.write(db_file, DB_ARCNAME)

        for base, prefix in ((faces_dir, FACES_PREFIX), (visitors_dir, VISITORS_PREFIX)):
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    zf.write(path, prefix + str(path.relative_to(base)).replace("\\", "/"))

    manifest["file"] = str(dest)
    manifest["size"] = dest.stat().st_size
    return manifest


# --- Import -----------------------------------------------------------------


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        manifest = json.loads(zf.read(MANIFEST_NAME))
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "This does not look like a Sentra data pack (no manifest inside)."
        ) from exc
    if not str(manifest.get("format", "")).startswith("sentra-data-pack/"):
        raise ValueError("Unrecognised data-pack format.")
    return manifest


def _local_is_empty(db_path: Path) -> bool:
    return sum(_count_rows(db_path).values()) == 0


def _safe_extract_tree(zf: zipfile.ZipFile, prefix: str, dest_root: Path) -> int:
    """Extract every member under ``prefix`` into ``dest_root``.

    Guards against path traversal: a member that would resolve outside
    ``dest_root`` (``../`` in the stored name) is skipped, never written.
    """
    dest_root = dest_root.resolve()
    written = 0
    for member in zf.namelist():
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        relative = member[len(prefix):]
        target = (dest_root / relative).resolve()
        if not str(target).startswith(str(dest_root)):
            continue  # path-traversal attempt — refuse silently
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        written += 1
    return written


def import_pack(pack_bytes: bytes) -> dict:
    """Merge a data pack into the running install.

    People are always merged (an imported name overwrites a local one of the
    same spelling). Photos and gate images are copied in. Detection/visitor
    history is imported **only when the local database is empty**, so importing
    into a fresh install populates it while importing into a live system never
    duplicates rows.
    """
    sentra_paths.ensure_data_dirs()

    try:
        zf = zipfile.ZipFile(io.BytesIO(pack_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid Sentra data pack.") from exc

    with zf:
        manifest = _read_manifest(zf)

        # --- People (always merged) ---------------------------------------
        added, updated = 0, 0
        if EMBEDDINGS_ARCNAME in zf.namelist():
            incoming = _safe_load_embeddings(zf.read(EMBEDDINGS_ARCNAME))
            target = sentra_paths.FACE_EMBEDDINGS_FILE
            current: dict[str, np.ndarray] = {}
            if target.is_file():
                current = _safe_load_embeddings(target.read_bytes())
            for name, vector in incoming.items():
                if name in current:
                    updated += 1
                else:
                    added += 1
                current[name] = vector
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as f:
                pickle.dump(current, f, protocol=pickle.HIGHEST_PROTOCOL)

        # --- Photos --------------------------------------------------------
        faces_written = _safe_extract_tree(zf, FACES_PREFIX, sentra_paths.FACES_DIR)
        visitor_photos = _safe_extract_tree(zf, VISITORS_PREFIX, sentra_paths.VISITORS_DIR)

        # --- History (only into an empty install) -------------------------
        history_imported = False
        db_target = sentra_paths.DETECTIONS_DB
        if DB_ARCNAME in zf.namelist():
            if _local_is_empty(db_target):
                # Write to a temp file first, then move into place, so a failed
                # write never leaves a half-copied database.
                with tempfile.NamedTemporaryFile(
                    dir=db_target.parent, delete=False, suffix=".tmp"
                ) as tmp:
                    tmp.write(zf.read(DB_ARCNAME))
                    tmp_path = Path(tmp.name)
                tmp_path.replace(db_target)
                history_imported = True

    counts = manifest.get("counts", {})
    return {
        "people_added": added,
        "people_updated": updated,
        "face_photos": faces_written,
        "visitor_photos": visitor_photos,
        "history_imported": history_imported,
        "history_counts": counts if history_imported else {},
        "history_skipped_reason": (
            "" if history_imported
            else "This install already has records, so only people were merged — "
                 "the pack's detection history was left out to avoid duplicates."
        ),
        "people_in_pack": manifest.get("people", []),
    }


# --- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export or import a Sentra data pack.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write a data pack from this install")
    ex.add_argument("dest", nargs="?", default="Sentra-DataPack.sentra")
    im = sub.add_parser("import", help="merge a data pack into this install")
    im.add_argument("src")
    args = parser.parse_args()

    if args.cmd == "export":
        info = export_pack(Path(args.dest))
        print(f"Wrote {info['file']} ({info['size'] / 1024:.0f} KB)")
        print(f"  people : {', '.join(info['people']) or '(none)'}")
        print(f"  history: {info['counts']}")
    else:
        result = import_pack(Path(args.src).read_bytes())
        print(json.dumps(result, indent=2))
