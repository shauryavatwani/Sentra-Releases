import json
import sqlite3
import os
from datetime import datetime, timedelta

from sentra_paths import DETECTIONS_DB

# Resolved centrally: on an installed Windows build the program folder is
# read-only, so the database has to live under ProgramData instead of next to
# the code. See sentra_paths for the full rationale.
DB_PATH = str(DETECTIONS_DB)

# Cooldown (seconds) to prevent duplicate entries
COOLDOWN = 30

# Stores the last time each (person, camera) was logged
last_logged = {}


def initialize_database():
    """
    Creates the SQLite database and its tables if they don't exist.
    """

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            camera_name TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    # Behavioural anomalies (currently only "fight", written by
    # anomaly_detection.py via face_recognition.py). `type` is a free text
    # column so future anomaly kinds don't need a schema migration.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            confidence REAL NOT NULL,
            persons_involved TEXT,
            bounding_boxes TEXT,
            camera_name TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def log_detection(person_name, camera_name):
    """
    Saves a detection event if it hasn't been logged recently.
    """

    current_time = datetime.now()

    key = (person_name, camera_name)

    if key in last_logged:
        if current_time - last_logged[key] < timedelta(seconds=COOLDOWN):
            return

    last_logged[key] = current_time

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections
        (person_name, camera_name, timestamp)
        VALUES (?, ?, ?)
    """, (
        person_name,
        camera_name,
        current_time.strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def log_anomaly(anomaly_type, confidence, persons_involved, bounding_boxes, camera_name):
    """
    Saves a behavioural anomaly (e.g. a detected fight).

    Unlike log_detection there is no cooldown here — the caller
    (anomaly_detection.py) already rate-limits per pair of people, and an
    anomaly is a rarer, higher-value event than a routine face sighting.

    persons_involved and bounding_boxes are stored as JSON text so a single
    row can describe an event involving any number of people.
    """

    current_time = datetime.now()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO anomalies
        (type, confidence, persons_involved, bounding_boxes, camera_name, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        anomaly_type,
        float(confidence),
        json.dumps(list(persons_involved)),
        json.dumps(list(bounding_boxes)),
        camera_name,
        current_time.strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# Automatically create the database when this file is imported
initialize_database()