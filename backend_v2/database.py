"""Read-only(ish) access to the real Database/detections.db written by event_logger.py.

Schema is not re-declared here — it's owned by Formal_Code/event_logger.py, which
creates the table on import. This module only queries it.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Harmless when frozen (the directory won't exist, but sentra_paths is then
# bundled as a top-level module and imports anyway).
sys.path.insert(0, str(PROJECT_ROOT / "Formal_Code"))
import sentra_paths  # noqa: E402  (path must be set up first)

# Imported for its import-time side effect: event_logger owns the `detections`
# and `anomalies` schema and creates both tables when loaded. Without this the
# backend could be the first process to touch a fresh install's database — it
# would find no tables and fail every stats poll until the engine happened to
# run. Schema ownership stays exactly where it was; this only guarantees the
# owner has run.
import event_logger  # noqa: E402,F401
import visitor_store  # noqa: E402

DB_PATH = sentra_paths.DETECTIONS_DB

# Alert type used for a visitor seen after their pass expired. Written by the
# engine into visitor_alerts (which has real columns for approved-until and
# how long overdue); mapped into the shared anomaly shape on read below.
VISITOR_OVERSTAY_TYPE = "visitor_overstay"


@contextlib.contextmanager
def _connect():
    """A connection that is committed *and closed* on the way out.

    Callers use ``with _connect() as conn``. Returning a bare Connection and
    letting the ``with`` handle it would only commit — ``sqlite3.Connection``
    does not close itself on ``__exit__``. CPython's refcounting happens to
    close it when the local goes out of scope, so this never misbehaved, but
    "correct because of when the garbage collector runs" is not a property to
    rely on for a file lock: on Windows an open handle blocks anything that
    needs to rewrite detections.db, and these reads run on a 4-second poll.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Every read below degrades to an empty result if the table is somehow missing,
# the same way the anomaly reads always have. An empty dashboard on a fresh
# machine is a fair description of reality; a 500 on every poll is not.


def recent_detections(limit: int = 10) -> list[dict]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, person_name, camera_name, timestamp "
                "FROM detections ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


def detections_today_count() -> int:
    today = date.today().isoformat()
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM detections WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["n"]) if row else 0


def detections_for_person(person_name: str, limit: int = 50) -> list[dict]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, person_name, camera_name, timestamp "
                "FROM detections WHERE person_name = ? ORDER BY id DESC LIMIT ?",
                (person_name, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


# --- Anomalies (fights) ----------------------------------------------------
# Written by Formal_Code/anomaly_detection.py through event_logger.log_anomaly.
# The table may not exist yet if the AI engine has never run on this machine,
# so the reads below degrade to an empty result instead of erroring.


def _anomaly_row_to_dict(row: sqlite3.Row) -> dict:
    """Expand the JSON text columns back into real lists."""
    record = dict(row)
    for field in ("persons_involved", "bounding_boxes"):
        raw = record.get(field)
        try:
            record[field] = json.loads(raw) if raw else []
        except (ValueError, TypeError):
            record[field] = []
    return record


def _overstay_as_anomaly(alert: dict) -> dict:
    """Present a visitor overstay in the same shape as a fight anomaly.

    The two are stored separately on purpose — an overstay is a fact with a
    duration ("approved until 15:00, seen at 16:20"), a fight is a scored
    guess — but an operator watching the Smart Alerts tab should not have to
    read two tables. Confidence is 1.0 because there is nothing probabilistic
    about a clock; the dashboard shows the overdue time instead of a percentage
    for this type rather than printing a meaningless "100% confident".
    """
    return {
        "id": f"visitor-{alert['id']}",
        "type": VISITOR_OVERSTAY_TYPE,
        "confidence": 1.0,
        "persons_involved": [alert["visitor_name"]],
        "bounding_boxes": [],
        "camera_name": alert["camera_name"],
        "timestamp": alert["detected_at"],
        # Extra fields a fight alert has no equivalent of; the dashboard uses
        # them for this type only.
        "visitor_id": alert["visitor_id"],
        "expires_at": alert["expires_at"],
        "overstay_seconds": alert["overstay_seconds"],
    }


def recent_anomalies(limit: int = 20, anomaly_type: str | None = None) -> list[dict]:
    """Fights and visitor overstays, newest first, in one feed."""
    fights: list[dict] = []
    if anomaly_type != VISITOR_OVERSTAY_TYPE:
        query = (
            "SELECT id, type, confidence, persons_involved, bounding_boxes, "
            "camera_name, timestamp FROM anomalies"
        )
        params: list = []
        if anomaly_type:
            query += " WHERE type = ?"
            params.append(anomaly_type)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        try:
            with _connect() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
            fights = [_anomaly_row_to_dict(row) for row in rows]
        except sqlite3.OperationalError:
            fights = []  # table not created yet — engine has never run

    overstays: list[dict] = []
    if anomaly_type in (None, VISITOR_OVERSTAY_TYPE):
        overstays = [
            _overstay_as_anomaly(a) for a in visitor_store.recent_visitor_alerts(limit)
        ]

    # Each source is already newest-first and capped at `limit`, so merging on
    # the timestamp and re-capping gives the correct newest-`limit` overall.
    merged = fights + overstays
    merged.sort(key=lambda row: row["timestamp"], reverse=True)
    return merged[:limit]


def anomalies_today_count() -> int:
    """Every alert raised today, of either kind."""
    today = date.today().isoformat()
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM anomalies WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()
        fights = int(row["n"]) if row else 0
    except sqlite3.OperationalError:
        fights = 0
    return fights + visitor_store.visitor_alerts_today_count()
