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

import contextlib
import datetime as _dt
import io
import json
import pickle
import shutil
import sqlite3
import sys
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


@contextlib.contextmanager
def _connect(db_path: Path):
    """A connection that is committed *and closed* on the way out.

    ``sqlite3.Connection`` is a context manager that only commits — it does
    not close. Leaving the handle open is survivable on POSIX and fatal on
    Windows, where an open handle blocks any attempt to replace or rewrite the
    database file. Same trap as ``visitor_store._connect``.
    """
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _count_rows(db_path: Path) -> dict[str, int]:
    counts = {t: 0 for t in HISTORY_TABLES}
    if not db_path.is_file():
        return counts
    with _connect(db_path) as conn:
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


# Characters Windows forbids in a filename, plus the device names it reserves
# whatever the extension. A pack written on macOS can legally contain any of
# these; opening one on Windows raises OSError, which would abort the whole
# import over a single badly-named photo.
_WINDOWS_FORBIDDEN_CHARS = set('<>:"|?*') | {chr(c) for c in range(32)}
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
}


def _is_writable_on_this_platform(relative: str) -> bool:
    if sys.platform != "win32":
        return True
    for part in Path(relative).parts:
        if set(part) & _WINDOWS_FORBIDDEN_CHARS:
            return False
        # Windows also strips trailing dots and spaces, so a name that ends in
        # one resolves to something other than what the pack recorded.
        if part != part.rstrip(" ."):
            return False
        if part.upper().split(".")[0] in _WINDOWS_RESERVED_NAMES:
            return False
    return True


def _safe_extract_tree(zf: zipfile.ZipFile, prefix: str, dest_root: Path) -> int:
    """Extract every member under ``prefix`` into ``dest_root``.

    Guards against path traversal: a member that would resolve outside
    ``dest_root`` (``../`` in the stored name) is skipped, never written.

    Also skips names this platform cannot represent, and treats a single file
    that will not write as a skip rather than a failure — losing one photo is
    a much better outcome than aborting an import of eight people's data.
    """
    dest_root = dest_root.resolve()
    written = 0
    for member in zf.namelist():
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        relative = member[len(prefix):]
        if not relative or not _is_writable_on_this_platform(relative):
            continue
        target = (dest_root / relative).resolve()
        if not str(target).startswith(str(dest_root)):
            continue  # path-traversal attempt — refuse silently
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except OSError:
            continue  # unwritable path, locked file, name too long
        written += 1
    return written


def _copy_history(db_bytes: bytes, db_target: Path) -> dict[str, int]:
    """Copy the pack's rows into the live database, table by table.

    Deliberately **not** a file replace. Replacing detections.db would need
    every handle on it closed, and this process is not the only one that opens
    it — the camera engine reads and writes it continuously. On Windows an open
    handle makes the replace fail outright (``PermissionError``), which is
    exactly how this surfaced: an HTTP 500 and "unexpected token 'I'" in the
    browser, because the error page is HTML and the caller expected JSON.

    Copying rows needs only a normal write transaction, so it works no matter
    who else has the file open. Columns are read from the source table rather
    than hard-coded, so a schema that gains a column later still imports.
    """
    imported: dict[str, int] = {}

    # The pack's database has to exist as a real file for sqlite to open it.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(db_bytes)
        source_path = Path(tmp.name)

    # Guarantee the destination schema before copying. event_logger owns
    # detections/anomalies and visitor_store owns visitors/visitor_alerts, and
    # both create their tables as an import side effect. Relying on "something
    # else will have imported these by now" is what made visitor rows import as
    # a silent zero: the copy hit "no such table", swallowed it as a
    # not-applicable table, and reported success having moved nothing.
    try:
        import event_logger  # noqa: F401  (imported for its table creation)
        import visitor_store  # noqa: F401
    except ImportError:
        pass

    try:
        with _connect(source_path) as src, _connect(db_target) as dst:
            for table in HISTORY_TABLES:
                try:
                    columns = [row[1] for row in src.execute(f"PRAGMA table_info({table})")]
                    if not columns:
                        continue
                    rows = src.execute(f"SELECT * FROM {table}").fetchall()
                except sqlite3.Error:
                    continue
                if not rows:
                    imported[table] = 0
                    continue

                placeholders = ",".join("?" * len(columns))
                column_list = ",".join(f'"{c}"' for c in columns)
                try:
                    # OR IGNORE so a re-import cannot raise on a colliding id;
                    # the empty-database guard above already makes that rare.
                    dst.executemany(
                        f'INSERT OR IGNORE INTO {table} ({column_list}) VALUES ({placeholders})',
                        rows,
                    )
                    imported[table] = len(rows)
                except sqlite3.Error:
                    # A table the live schema does not have (an older build
                    # importing a newer pack) is skipped, not fatal.
                    imported[table] = 0
    finally:
        source_path.unlink(missing_ok=True)

    return imported


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
        imported_counts: dict[str, int] = {}
        db_target = sentra_paths.DETECTIONS_DB
        if DB_ARCNAME in zf.namelist() and _local_is_empty(db_target):
            imported_counts = _copy_history(zf.read(DB_ARCNAME), db_target)
            history_imported = True

    # Report what actually landed, not what the manifest claimed was packed.
    counts = imported_counts or manifest.get("counts", {})
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
