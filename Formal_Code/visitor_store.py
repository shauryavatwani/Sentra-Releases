"""Temporary Pass — storage for gate visitors and their overstay alerts.

A visitor is someone who is *not* enrolled in the permanent face database
(``face_embeddings.pkl``). The security guard on the gate photographs them,
sets how long the visit should last, and management approves it. From that
moment the AI engine recognises them by name like anyone else — but only until
their window closes. Seen after that, they raise an overstay alert.

Why visitors live here and not in ``face_embeddings.pkl``
--------------------------------------------------------
Two reasons, both deliberate:

1. **The permanent database is permanent.** Staff and students are enrolled
   from curated photos and are meant to stay. A visitor is a single afternoon.
   Mixing them would make "registered individuals" a number that drifts upward
   every day and would leave dead embeddings behind forever.
2. **Visitors need a clock, and a pickle has nowhere to put one.** Approval
   time, expiry, who approved it, when access was ended early — that is a row,
   not a 512-float vector.

So the embedding maths is identical (same InsightFace pipeline, same
normalisation, same 0.45 cosine threshold — see face_recognition.py), but the
storage is a table alongside ``detections`` and ``anomalies`` in the same
``Database/detections.db``.

Tables owned by this module
---------------------------
``visitors``
    One row per visit request, including the 512-float embedding as a BLOB and
    the filename of the captured photo (the image itself lives under
    ``Visitors/``, the same way ``Faces/`` holds enrolled photos).

``visitor_alerts``
    One row per overstay alert. Kept separate from ``anomalies`` because an
    overstay is a *fact with a duration* (approved until X, seen at Y), not a
    scored guess like a fight. The backend maps these into the shared alert
    feed on read so the Smart Alerts tab still shows everything in one place.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np

from sentra_paths import DETECTIONS_DB, VISITORS_DIR

DB_PATH = str(DETECTIONS_DB)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

# How long after a visit expires the engine keeps watching for that face.
# Someone who wandered back in an hour after their pass ran out is exactly what
# this feature is for; someone recognised three days later is noise, and every
# retained visitor costs a slot in the matcher for every frame.
OVERSTAY_WATCH_HOURS = 24

# One alert per visitor per hour, as specified. A visitor standing in view of a
# camera is detected several times a second; without this the alerts table
# would fill with the same fact.
OVERSTAY_ALERT_COOLDOWN_SECONDS = 3600

# How long a visitor's gate photo survives after their pass itself expired
# (revoked visits count too — see purge_expired_photos). A photo of someone's
# face is the most sensitive thing this feature stores, and once the visit is
# over there is no ongoing reason to keep it — the visit record (name, times,
# who approved it) stays for the "who was here" history; only the image goes.
PHOTO_RETENTION_HOURS = 24

MIN_DURATION_MINUTES = 5
MAX_DURATION_MINUTES = 24 * 60

EMBEDDING_DIMENSIONS = 512

# Both the backend (several request threads) and the engine write here.
_lock = threading.RLock()


def _now() -> datetime:
    return datetime.now()


def _fmt(moment: datetime) -> str:
    return moment.strftime(TIME_FORMAT)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError:
        return None


@contextmanager
def _connect():
    """A connection that is committed *and closed* on the way out.

    ``sqlite3.Connection`` is itself a context manager, but it only commits —
    it does not close. That is survivable in a request handler and quietly
    fatal in the engine, which reloads the visitor list every few seconds
    forever and would accumulate file handles until it ran out.

    timeout: the engine and the backend genuinely do write concurrently (a
    detection landing while a guard submits a visitor), and retrying for a few
    seconds is far better than surfacing "database is locked".
    """
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize_database() -> None:
    """Create the visitor tables if they don't exist. Safe to call repeatedly."""
    DETECTIONS_DB.parent.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visitors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                purpose TEXT,
                host TEXT,
                status TEXT NOT NULL,
                embedding BLOB NOT NULL,
                photo_file TEXT,
                duration_minutes INTEGER NOT NULL,
                requested_by TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                decided_by TEXT,
                decided_at TEXT,
                approved_at TEXT,
                expires_at TEXT,
                rejection_reason TEXT,
                revoked_at TEXT,
                revoked_by TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS visitor_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT NOT NULL,
                visitor_name TEXT NOT NULL,
                camera_name TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                overstay_seconds INTEGER NOT NULL
            )
        """)

        # The engine reloads the approved set every few seconds and the alert
        # cooldown is checked per sighting, so both of those get an index.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visitors_status ON visitors(status)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visitor_alerts_visitor "
            "ON visitor_alerts(visitor_id, id DESC)"
        )


# --- Embeddings -------------------------------------------------------------


def encode_embedding(embedding: np.ndarray) -> bytes:
    """Validate and pack a 512-float embedding for storage.

    Validated here rather than trusted, because a wrong-shaped array would not
    fail until it reached the matcher inside the engine — at which point the
    visitor silently never matches and the guard has no idea why.
    """
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (EMBEDDING_DIMENSIONS,):
        raise ValueError(
            f"Expected a {EMBEDDING_DIMENSIONS}-value embedding, got {vector.shape[0]}."
        )
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Refusing to store a zero-length embedding.")
    return (vector / norm).astype(np.float32).tobytes()


def decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(-1)


# --- Reading ----------------------------------------------------------------


def visit_state(row: sqlite3.Row | dict) -> str:
    """The state a human cares about, derived rather than stored.

    Storing this would mean something had to write "expired" into every row the
    moment its clock ran out — a background job that can silently stop. Derived
    from the timestamps, expiry is simply always correct.
    """
    status = row["status"]
    if status != STATUS_APPROVED:
        return status  # pending / rejected

    expires_at = parse_time(row["expires_at"])
    if expires_at is None:
        return STATUS_APPROVED
    if _now() < expires_at:
        return "on_premises"
    return "ended_early" if row["revoked_at"] else "overdue"


def public_visitor(row: sqlite3.Row | dict) -> dict:
    """The API-safe view: everything except the raw embedding bytes."""
    record = dict(row)
    record.pop("embedding", None)
    record["state"] = visit_state(row)
    record["has_photo"] = bool(record.get("photo_file"))

    expires_at = parse_time(record.get("expires_at"))
    if expires_at is None:
        record["seconds_remaining"] = None
    else:
        record["seconds_remaining"] = int((expires_at - _now()).total_seconds())
    return record


def get_visitor(visitor_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM visitors WHERE id = ?", (visitor_id,)).fetchone()
    return dict(row) if row else None


def list_visitors(status: str | None = None, limit: int = 200) -> list[dict]:
    query = "SELECT * FROM visitors"
    params: list = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY requested_at DESC, rowid DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def visitors_for_matching() -> list[dict]:
    """Approved visitors the engine should currently be able to recognise.

    Includes visitors whose window has already closed — that is the whole point,
    since an expired visitor still in the building is what raises the alert.
    They drop out only once they are past OVERSTAY_WATCH_HOURS.
    """
    cutoff = _fmt(_now() - timedelta(hours=OVERSTAY_WATCH_HOURS))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, embedding, expires_at FROM visitors "
            "WHERE status = ? AND (expires_at IS NULL OR expires_at > ?)",
            (STATUS_APPROVED, cutoff),
        ).fetchall()

    visitors = []
    for row in rows:
        try:
            vector = decode_embedding(row["embedding"])
        except ValueError:
            continue
        if vector.shape != (EMBEDDING_DIMENSIONS,):
            continue
        visitors.append(
            {
                "id": row["id"],
                "name": row["name"],
                "embedding": vector,
                "expires_at": row["expires_at"],
            }
        )
    return visitors


def counts_by_state() -> dict:
    """Headline numbers for the dashboard, computed in one pass."""
    rows = list_visitors(limit=1000)
    tally = {"pending": 0, "on_premises": 0, "overdue": 0}
    for row in rows:
        state = visit_state(row)
        if state in tally:
            tally[state] += 1
    return tally


# --- Writing ----------------------------------------------------------------


def _new_id() -> str:
    return f"vis_{secrets.token_hex(4)}"


def validate_duration(minutes: int) -> int:
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        raise ValueError("Visit duration must be a whole number of minutes.") from None
    if minutes < MIN_DURATION_MINUTES:
        raise ValueError(f"Visit duration must be at least {MIN_DURATION_MINUTES} minutes.")
    if minutes > MAX_DURATION_MINUTES:
        raise ValueError("Visit duration cannot exceed 24 hours.")
    return minutes


def create_visitor(
    *,
    name: str,
    embedding: np.ndarray,
    photo_file: str,
    duration_minutes: int,
    requested_by: str,
    purpose: str = "",
    host: str = "",
) -> dict:
    """Record a visit request. Always starts as pending — the clock does not
    begin until someone with authority approves it."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Visitor name is required.")

    duration_minutes = validate_duration(duration_minutes)
    blob = encode_embedding(embedding)

    visitor_id = _new_id()
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO visitors
            (id, name, purpose, host, status, embedding, photo_file,
             duration_minutes, requested_by, requested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                visitor_id,
                name,
                purpose.strip(),
                host.strip(),
                STATUS_PENDING,
                blob,
                photo_file,
                duration_minutes,
                requested_by,
                _fmt(_now()),
            ),
        )
    return get_visitor(visitor_id)


def set_photo_file(visitor_id: str, photo_file: str) -> None:
    """Attach the saved photo filename, which is only known after the row
    exists (the file is named after the generated visitor id)."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE visitors SET photo_file = ? WHERE id = ?", (photo_file, visitor_id)
        )


def _visitors_with_photos_to_purge() -> list[dict]:
    """Approved visits whose pass closed more than PHOTO_RETENTION_HOURS ago
    and still have a photo on disk.

    ``expires_at`` covers this regardless of *how* the visit ended: it is set
    the moment access started (approval), and moved to "now" on an early
    revoke (see revoke_visitor) — so a revoked visit's photo ages out exactly
    PHOTO_RETENTION_HOURS after the revoke, not after the original approval.
    """
    cutoff = _fmt(_now() - timedelta(hours=PHOTO_RETENTION_HOURS))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, photo_file FROM visitors "
            "WHERE status = ? AND photo_file != '' AND expires_at IS NOT NULL "
            "AND expires_at < ?",
            (STATUS_APPROVED, cutoff),
        ).fetchall()
    return [dict(row) for row in rows]


def purge_expired_photos() -> int:
    """Delete gate photos whose pass expired more than PHOTO_RETENTION_HOURS ago.

    Only the image file and the ``photo_file`` column are cleared — the visit
    record itself (name, times, who approved it) is left alone, since "who was
    on the premises and when" is exactly the history an incident enquiry would
    ask for. Safe to call repeatedly (a missing file is not an error) and safe
    to run from more than one process at once.

    Returns how many photos were purged, for the caller to log.
    """
    purged = 0
    for row in _visitors_with_photos_to_purge():
        path = VISITORS_DIR / row["photo_file"]
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Leave photo_file in place and retry on the next sweep rather
            # than losing track of a file that may still exist.
            continue
        with _lock, _connect() as conn:
            conn.execute(
                "UPDATE visitors SET photo_file = '' WHERE id = ?", (row["id"],)
            )
        purged += 1
    return purged


def approve_visitor(visitor_id: str, approved_by: str) -> dict:
    """Approve a pending request; the visit clock starts now.

    Deliberately measured from approval rather than from when the guard filled
    the form: the visitor is standing at the gate waiting for the decision, so
    a delay in approving must not eat into their time inside.
    """
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM visitors WHERE id = ?", (visitor_id,)).fetchone()
        if row is None:
            raise KeyError(visitor_id)
        if row["status"] != STATUS_PENDING:
            raise ValueError(f"This request has already been {row['status']}.")

        now = _now()
        expires = now + timedelta(minutes=int(row["duration_minutes"]))
        conn.execute(
            "UPDATE visitors SET status = ?, decided_by = ?, decided_at = ?, "
            "approved_at = ?, expires_at = ? WHERE id = ?",
            (STATUS_APPROVED, approved_by, _fmt(now), _fmt(now), _fmt(expires), visitor_id),
        )
    return get_visitor(visitor_id)


def reject_visitor(visitor_id: str, rejected_by: str, reason: str = "") -> dict:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT status FROM visitors WHERE id = ?", (visitor_id,)).fetchone()
        if row is None:
            raise KeyError(visitor_id)
        if row["status"] != STATUS_PENDING:
            raise ValueError(f"This request has already been {row['status']}.")

        conn.execute(
            "UPDATE visitors SET status = ?, decided_by = ?, decided_at = ?, "
            "rejection_reason = ? WHERE id = ?",
            (STATUS_REJECTED, rejected_by, _fmt(_now()), reason.strip(), visitor_id),
        )
    return get_visitor(visitor_id)


def extend_visitor(visitor_id: str, extra_minutes: int, extended_by: str) -> dict:
    """Give an approved visitor more time.

    Extending an already-overdue visit measures from now, not from the original
    expiry — otherwise granting "30 more minutes" to someone two hours overdue
    would hand them a pass that is still expired, and they would keep alerting.
    """
    extra_minutes = validate_duration(extra_minutes)

    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM visitors WHERE id = ?", (visitor_id,)).fetchone()
        if row is None:
            raise KeyError(visitor_id)
        if row["status"] != STATUS_APPROVED:
            raise ValueError("Only an approved visit can be extended.")
        if not row["photo_file"]:
            # Same call path handles both "Extend" (still active — the photo
            # is always present, since purge only runs after expiry) and
            # "Re-issue" (already finished — the gate photo may have aged out
            # via purge_expired_photos). Once the photo is gone there is
            # nothing left to re-verify this person against, so re-issuing
            # must raise a fresh visit request instead of reviving this one.
            raise ValueError(
                "This visitor's gate photo has been deleted after the retention "
                "window closed. Raise a new visit request instead of re-issuing this one."
            )

        now = _now()
        current_expiry = parse_time(row["expires_at"]) or now
        base = max(current_expiry, now)
        new_expiry = base + timedelta(minutes=extra_minutes)

        conn.execute(
            "UPDATE visitors SET expires_at = ?, duration_minutes = ?, "
            "revoked_at = NULL, revoked_by = NULL WHERE id = ?",
            (
                _fmt(new_expiry),
                int(row["duration_minutes"]) + extra_minutes,
                visitor_id,
            ),
        )
    return get_visitor(visitor_id)


def revoke_visitor(visitor_id: str, revoked_by: str) -> dict:
    """End a visit immediately.

    This does not delete the visitor or stop watching for them — it sets the
    expiry to now, which means they are monitored exactly like anyone whose
    pass ran out. Someone whose access was pulled is precisely the person you
    still want to be alerted about, so removing them from the matcher would be
    the wrong reading of "revoke".
    """
    with _lock, _connect() as conn:
        row = conn.execute("SELECT status FROM visitors WHERE id = ?", (visitor_id,)).fetchone()
        if row is None:
            raise KeyError(visitor_id)
        if row["status"] != STATUS_APPROVED:
            raise ValueError("Only an approved visit can be ended.")

        now = _fmt(_now())
        conn.execute(
            "UPDATE visitors SET expires_at = ?, revoked_at = ?, revoked_by = ? WHERE id = ?",
            (now, now, revoked_by, visitor_id),
        )
    return get_visitor(visitor_id)


def delete_visitor(visitor_id: str) -> bool:
    """Remove a visit record entirely, along with its alerts."""
    with _lock, _connect() as conn:
        removed = conn.execute("DELETE FROM visitors WHERE id = ?", (visitor_id,)).rowcount
        conn.execute("DELETE FROM visitor_alerts WHERE visitor_id = ?", (visitor_id,))
    return removed > 0


# --- Overstay alerts --------------------------------------------------------


def seconds_since_last_alert(visitor_id: str) -> float | None:
    """Age of this visitor's most recent overstay alert, or None if never."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT detected_at FROM visitor_alerts WHERE visitor_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (visitor_id,),
        ).fetchone()
    if row is None:
        return None
    last = parse_time(row["detected_at"])
    if last is None:
        return None
    return (_now() - last).total_seconds()


def log_overstay(
    visitor_id: str, visitor_name: str, camera_name: str, expires_at: str
) -> dict | None:
    """Record an overstay sighting, respecting the one-per-hour rule.

    Returns the alert when one was written, or None when it was suppressed by
    the cooldown. The cooldown is checked against the database rather than an
    in-memory timer so that restarting the engine does not immediately re-alert
    on every overdue visitor still in the building.
    """
    age = seconds_since_last_alert(visitor_id)
    if age is not None and age < OVERSTAY_ALERT_COOLDOWN_SECONDS:
        return None

    now = _now()
    expiry = parse_time(expires_at)
    overstay_seconds = int((now - expiry).total_seconds()) if expiry else 0

    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO visitor_alerts
            (visitor_id, visitor_name, camera_name, detected_at, expires_at, overstay_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (visitor_id, visitor_name, camera_name, _fmt(now), expires_at, overstay_seconds),
        )

    return {
        "visitor_id": visitor_id,
        "visitor_name": visitor_name,
        "camera_name": camera_name,
        "detected_at": _fmt(now),
        "expires_at": expires_at,
        "overstay_seconds": overstay_seconds,
    }


def recent_visitor_alerts(limit: int = 50) -> list[dict]:
    """Overstay alerts, newest first, joined to the visitor for the photo."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*, v.photo_file, v.approved_at, v.host, v.purpose
                FROM visitor_alerts a
                LEFT JOIN visitors v ON v.id = a.visitor_id
                ORDER BY a.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.OperationalError:
        return []  # tables not created yet on this machine
    return [dict(row) for row in rows]


def visitor_alerts_today_count() -> int:
    today = _now().strftime("%Y-%m-%d")
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM visitor_alerts WHERE detected_at LIKE ?",
                (f"{today}%",),
            ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["n"]) if row else 0


# Created on import, matching event_logger.py's behaviour so that whichever
# process touches the database first — engine or backend — finds it ready.
initialize_database()
