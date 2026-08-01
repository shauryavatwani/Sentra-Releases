"""Read-only(ish) access to the real Database/detections.db written by event_logger.py.

Schema is not re-declared here — it's owned by Formal_Code/event_logger.py, which
creates the table on import. This module only queries it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "Database" / "detections.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def recent_detections(limit: int = 10) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, person_name, camera_name, timestamp "
            "FROM detections ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def detections_today_count() -> int:
    today = date.today().isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM detections WHERE timestamp LIKE ?",
            (f"{today}%",),
        ).fetchone()
    return int(row["n"]) if row else 0


def detections_for_person(person_name: str, limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, person_name, camera_name, timestamp "
            "FROM detections WHERE person_name = ? ORDER BY id DESC LIMIT ?",
            (person_name, limit),
        ).fetchall()
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


def recent_anomalies(limit: int = 20, anomaly_type: str | None = None) -> list[dict]:
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
    except sqlite3.OperationalError:
        return []  # table not created yet — engine has never run
    return [_anomaly_row_to_dict(row) for row in rows]


def anomalies_today_count() -> int:
    today = date.today().isoformat()
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM anomalies WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["n"]) if row else 0
