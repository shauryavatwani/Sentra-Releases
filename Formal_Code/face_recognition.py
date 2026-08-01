"""Recognize enrolled, consented test participants from the RTSP camera feed(s).

Runs one capture + streaming pipeline per configured camera (camera_store.py).
Every camera streams its live feed to the dashboard; face recognition and
fight detection additionally run on whichever cameras have `ai_enabled` set —
that split exists because each one costs a full InsightFace + pose-model pass
per frame, and not every deployment can afford that on every camera.

One camera failing to open (wrong IP, powered off) does not stop the others —
each camera's pipeline is independent and reports its own failure.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
import threading
import time
from datetime import datetime

import camera_store
from event_logger import log_detection, log_anomaly
import cv2
import numpy as np
import websocket
from insightface.app import FaceAnalysis

import anomaly_detection
import sentra_paths
import visitor_store


# ----------------------------- Configuration -----------------------------

DATABASE_FILE = sentra_paths.FACE_EMBEDDINGS_FILE

# A higher value reduces mistaken matches but may label more real people as
# Unknown. Tune this only using consenting test participants.
MATCH_THRESHOLD = 0.45
AI_FRAME_SIZE = (480, 270)  # width, height processed by the AI worker
AI_INTERVAL_SECONDS = 0.25  # about four recognition attempts per second

# Dashboard streaming (backend_v2/main.py relays this to any open dashboard).
# Streaming is best-effort: if a backend isn't running, that one target
# quietly retries in the background — it never blocks the local preview
# window, and other targets/cameras are unaffected. Stayed a list because the
# streaming code fans out over it — a second entry is all it takes if another
# dashboard is ever added.
DASHBOARD_WS_URLS = [
    "ws://localhost:8000/ws/engine",
]
STREAM_INTERVAL_SECONDS = 0.15  # ~6.6fps target; 0.1 was jittery (CPU-bound), 0.3 felt laggy
STREAM_MAX_WIDTH = 720  # smaller frames offset the higher send rate
STREAM_JPEG_QUALITY = 60

# --- Local preview windows --------------------------------------------------
# Running from source, cv2.imshow windows are a genuinely useful development
# view. In an installed build they are wrong in three separate ways:
#
#   * The dashboard's Live Monitor is the product's UI. The engine is a
#     background service started by the launcher; desktop windows appearing
#     from a process the user never launched read as a malfunction.
#   * The engine is spawned DETACHED_PROCESS with no console, so there is no
#     obvious way to close those windows, and the "press Q to quit" instruction
#     printed alongside them goes to a log file nobody is reading.
#   * highgui is the one part of OpenCV that can fail purely on environment
#     (no window station, a headless build). A raise there would take down face
#     recognition, which has nothing to do with drawing a preview.
#
# So: preview from source, headless once frozen. SENTRA_PREVIEW forces it
# either way ("1" on, "0" off) for debugging an installed build.
def _preview_enabled() -> bool:
    override = os.environ.get("SENTRA_PREVIEW")
    if override is not None:
        return override.strip() not in ("", "0", "false", "no")
    return not getattr(sys, "frozen", False)


SHOW_PREVIEW = _preview_enabled()
# cv2.waitKey is what paces the display loop; with no window to wait on, the
# loop would spin as fast as the stream can hand back frames and burn a core
# for nothing. Headless runs sleep for the equivalent interval instead.
HEADLESS_LOOP_SLEEP = 0.05

# Fight/anomaly detection (Formal_Code/anomaly_detection.py).
# The pose model is the most expensive thing in this script after InsightFace,
# so it runs on every Nth recognition cycle rather than every one. A fight
# lasts seconds and the classifier needs several samples to fire anyway, so
# ~2 evaluations/sec is plenty while roughly halving the added CPU cost.
ANOMALY_DETECTION_ENABLED = True
ANOMALY_EVERY_N_CYCLES = 2
ANOMALY_HISTORY_LENGTH = 20  # recent anomalies kept for streaming to dashboards

# Temporary Pass (visitors). Unlike the enrolled database, which is read once at
# startup, the visitor set changes while the engine runs — a guard signs someone
# in at the gate and they must be recognised seconds later, not after a restart.
VISITOR_RELOAD_SECONDS = 5

# The enrolled database changes too — someone is registered in the dashboard,
# or a data pack is imported. Both used to need an engine restart to take
# effect, with nothing in the UI saying so. Checked by modification time, so a
# tick where nothing changed costs one stat() call; 5s matches the visitor
# registry rather than being tuned separately.
ENROLLED_RELOAD_SECONDS = 5


class LatestFrameStream:
    """Continuously reads RTSP frames and keeps only the newest one."""

    def __init__(self, url: str) -> None:
        self._capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._capture.isOpened():
            raise RuntimeError("Could not open the RTSP camera stream.")
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._capture.release()


class VisitorRegistry:
    """The currently-recognisable visitors, kept fresh while the engine runs.

    The enrolled database is loaded once at startup because it changes rarely
    and a restart is an acceptable cost. Visitors are the opposite: a guard
    photographs someone at the gate, management approves, and the visitor walks
    in *immediately*. Waiting for an engine restart would make the feature
    useless, so this reloads on a timer in the background.

    Reads never lock. The loader thread builds a complete new snapshot and
    swaps it into ``_snapshot`` in one assignment, so a recognition worker
    either sees the whole old set or the whole new one — never a half-updated
    matrix whose rows and names disagree.

    One registry is shared by every camera. Unlike the pose tracker (which
    holds per-camera ByteTrack state and must not be shared), this is read-only
    reference data with nothing to corrupt.
    """

    def __init__(self) -> None:
        # (ids, names, expiries, matrix) or None when nobody is signed in.
        self._snapshot: tuple[list[str], list[str], list[str], np.ndarray] | None = None
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def reload(self) -> int:
        """Rebuild the snapshot from the database. Returns the visitor count."""
        try:
            visitors = visitor_store.visitors_for_matching()
        except Exception as exc:
            # A failed reload must not kill recognition — the previous snapshot
            # stays in place and the next tick tries again.
            self.last_error = str(exc)
            return -1

        self.last_error = None
        if not visitors:
            self._snapshot = None
            return 0

        matrix = np.stack([v["embedding"] for v in visitors])
        self._snapshot = (
            [v["id"] for v in visitors],
            [v["name"] for v in visitors],
            [v["expires_at"] for v in visitors],
            matrix,
        )
        return len(visitors)

    def _loop(self, running: threading.Event) -> None:
        previous = -1
        while running.is_set():
            count = self.reload()
            # Only speak up when the number actually changes, so the log stays
            # readable rather than emitting a line every five seconds forever.
            if count >= 0 and count != previous:
                print(f"Temporary Pass: {count} visitor(s) currently recognisable.")
                previous = count
            time.sleep(VISITOR_RELOAD_SECONDS)

    def start(self, running: threading.Event) -> None:
        self.reload()
        self._thread = threading.Thread(target=self._loop, args=(running,), daemon=True)
        self._thread.start()

    def best_match(self, embedding: np.ndarray) -> dict | None:
        """Closest visitor above the match threshold, or None.

        Uses the same normalised cosine similarity and the same
        MATCH_THRESHOLD as the enrolled database — a visitor is not recognised
        on looser evidence than anyone else.
        """
        snapshot = self._snapshot  # single read; see class docstring
        if snapshot is None:
            return None

        ids, names, expiries, matrix = snapshot
        normalized = embedding / np.linalg.norm(embedding)
        scores = matrix @ normalized
        index = int(np.argmax(scores))
        score = float(scores[index])
        if score < MATCH_THRESHOLD:
            return None
        return {
            "id": ids[index],
            "name": names[index],
            "expires_at": expiries[index],
            "score": score,
        }


def identify(
    embedding: np.ndarray,
    names: list[str],
    known_embeddings: np.ndarray,
    registry: VisitorRegistry | None,
) -> tuple[str, float, dict | None]:
    """Match one face against enrolled people and current visitors.

    Both sets are scored and the higher similarity wins, rather than checking
    visitors only when the enrolled lookup fails. A weak-but-above-threshold
    match against an enrolled person should not beat a strong match against the
    visitor standing in front of the camera.
    """
    label, score = best_match(embedding, names, known_embeddings)

    visitor = registry.best_match(embedding) if registry is not None else None
    if visitor is not None and visitor["score"] > score:
        return visitor["name"], visitor["score"], visitor
    return label, score, None


def visitor_is_overdue(visitor: dict) -> bool:
    expires_at = visitor_store.parse_time(visitor.get("expires_at"))
    return expires_at is not None and datetime.now() > expires_at


class CameraRuntime:
    """All the mutable state one camera's pipeline needs, gathered in one place.

    Kept as an object rather than a pile of dicts-per-camera-id so that
    per-camera locks stay correctly scoped: each camera has its own
    results_lock/anomalies_lock, never shared with any other camera.
    """

    def __init__(self, config: dict) -> None:
        self.id: str = config["id"]
        self.name: str = config["name"]
        self.rtsp_url: str = config["rtsp_url"]
        self.ai_enabled: bool = bool(config["ai_enabled"])

        self.stream: LatestFrameStream | None = None
        self.online = False  # stream opened successfully

        # A FightDetector is per-camera and owns its own YOLO model instance —
        # see anomaly_detection.py's module docstring for why sharing one
        # across cameras would corrupt ByteTrack ids across cameras.
        self.detector: anomaly_detection.FightDetector | None = None

        self.latest_results: list[tuple[int, int, int, int, str, float]] = []
        self.results_lock = threading.Lock()

        # Anomalies are events rather than state: each one gets an increasing
        # event_id so every dashboard stream can send each event exactly once
        # instead of re-alerting on the same fight every frame.
        self.latest_anomalies: list[dict] = []
        self.anomalies_lock = threading.Lock()
        self.next_anomaly_event_id = 0

    def publish_anomalies(self, anomalies: list[dict]) -> None:
        """Queue events for the dashboard stream, stamping each with an id.

        The increasing event_id is what lets every connected dashboard show an
        event exactly once instead of re-alerting on every frame for as long as
        the situation lasts. Used by both fight detection and visitor
        overstays, so the two cannot drift apart in how they are delivered.
        """
        if not anomalies:
            return
        with self.anomalies_lock:
            for anomaly in anomalies:
                anomaly["event_id"] = self.next_anomaly_event_id
                anomaly.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                self.next_anomaly_event_id += 1
                self.latest_anomalies.append(anomaly)
            del self.latest_anomalies[:-ANOMALY_HISTORY_LENGTH]


def _empty_database() -> tuple[list[str], np.ndarray]:
    """No enrolled people — a valid state, not an error.

    A freshly installed Sentra has nobody registered yet. Everything except
    naming an enrolled face still works: the camera streams to Live Monitor,
    fight detection runs, visitors on a Temporary Pass are recognised, and
    unrecognised people are reported as Unknown. Treating this as fatal would
    mean a new install starts with a dead engine and a Live Monitor that never
    comes up, before the user has had any chance to register anyone.

    The zero-row matrix keeps the shape contract (N, 512) so the matching code
    needs no special case: a dot product against it yields no matches, which is
    exactly the right answer.
    """
    return [], np.zeros((0, 512), dtype=np.float32)


def load_database() -> tuple[list[str], np.ndarray]:
    """Load and validate the locally generated face-embedding database.

    Never fatal. A missing or unreadable database degrades to "nobody is
    enrolled" and says so in the log, because the engine has plenty of work
    left to do without it.
    """
    if not DATABASE_FILE.is_file():
        print(
            "No enrolled people yet — the face database does not exist. "
            "Cameras, fight detection and Temporary Pass all still run; "
            "everyone will be reported as Unknown until someone is registered "
            "in the dashboard (Register person), or a data pack is imported."
        )
        return _empty_database()

    try:
        with DATABASE_FILE.open("rb") as database_handle:
            raw_database = pickle.load(database_handle)
    except Exception as exc:  # noqa: BLE001 — a corrupt pickle must not be fatal
        print(
            f"Could not read the face database ({exc}). Continuing with nobody "
            "enrolled; re-register or re-import to repair it."
        )
        return _empty_database()

    if not isinstance(raw_database, dict):
        print("The face database has an unexpected format; continuing with nobody enrolled.")
        return _empty_database()
    if not raw_database:
        print("The face database is empty; everyone will be reported as Unknown.")
        return _empty_database()

    names: list[str] = []
    embeddings: list[np.ndarray] = []
    skipped: list[str] = []
    for name, embedding in raw_database.items():
        # One malformed row must not cost the other seven people their
        # recognition — skip it and carry on, loudly.
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            skipped.append(str(name))
            continue
        if vector.shape != (512,) or np.linalg.norm(vector) == 0:
            skipped.append(str(name))
            continue
        names.append(str(name))
        embeddings.append(vector / np.linalg.norm(vector))

    if skipped:
        print(f"Skipped {len(skipped)} unusable embedding(s): {', '.join(skipped)}")
    if not embeddings:
        print("No usable embeddings in the face database; everyone will be Unknown.")
        return _empty_database()

    return names, np.stack(embeddings)


class EnrolledRegistry:
    """The enrolled people, reloaded whenever their database file changes.

    This used to be read exactly once, at engine startup, and that was wrong in
    a way that only shows up in a packaged build. A shipped Sentra starts with
    nobody enrolled — the installer carries no biometric data — so the very
    first thing an operator does is register someone or import a data pack.
    With a startup-only read, neither took effect until the engine was
    restarted, and nothing in the UI said so. The symptom is the worst kind:
    the camera streams, the dashboard is healthy, detection simply never names
    anybody, and there is no error anywhere to explain it.

    Reload is keyed on the file's modification time, so the common case costs
    one stat() per tick and nothing else. Reads never lock: the loader builds a
    complete snapshot and swaps it in with one assignment, so a worker sees
    either the whole old roster or the whole new one — never a matrix whose
    rows and names disagree. Same contract as VisitorRegistry, for the same
    reason.
    """

    def __init__(self) -> None:
        self._snapshot: tuple[list[str], np.ndarray] = _empty_database()
        self._signature: tuple[float, int] | None = None
        self._thread: threading.Thread | None = None

    def _file_signature(self) -> tuple[float, int] | None:
        try:
            stat = DATABASE_FILE.stat()
        except OSError:
            return None
        # Size as well as mtime: two writes inside the same coarse mtime tick
        # is unlikely but free to rule out.
        return (stat.st_mtime, stat.st_size)

    def reload_if_changed(self) -> bool:
        """Reload when the file has changed. Returns True if it did."""
        signature = self._file_signature()
        if signature == self._signature:
            return False
        self._signature = signature
        try:
            self._snapshot = load_database()
        except Exception as exc:  # noqa: BLE001 — never kill recognition
            print(f"Could not reload the face database ({exc}); keeping the previous set.")
            return False
        return True

    def get(self) -> tuple[list[str], np.ndarray]:
        return self._snapshot  # single read; see class docstring

    def _loop(self, running: threading.Event) -> None:
        while running.is_set():
            time.sleep(ENROLLED_RELOAD_SECONDS)
            if self.reload_if_changed():
                names, _ = self._snapshot
                print(
                    f"Face database updated — now {len(names)} enrolled person(s)"
                    f"{': ' + ', '.join(names) if names else ''}"
                )

    def start(self, running: threading.Event) -> None:
        self.reload_if_changed()
        self._thread = threading.Thread(
            target=self._loop, args=(running,), name="enrolled-reload", daemon=True
        )
        self._thread.start()


def load_model() -> FaceAnalysis:
    print("Loading InsightFace model...")
    # root= points InsightFace at the model set. A packaged build ships
    # buffalo_l inside the install folder so the client PC needs no internet on
    # first run; from source this resolves to the usual ~/.insightface cache.
    model = FaceAnalysis(
        root=str(sentra_paths.insightface_root()),
        providers=["CPUExecutionProvider"],
    )
    model.prepare(ctx_id=0, det_size=(320, 320))
    print("Model loaded.")
    return model


def best_match(
    embedding: np.ndarray, names: list[str], known_embeddings: np.ndarray
) -> tuple[str, float]:
    """Return the best cosine-similarity match, or Unknown below threshold."""
    # Nobody enrolled yet (a fresh install — see load_database). np.argmax
    # raises on an empty sequence, so without this the engine would survive
    # startup and then die the first time a face appeared, which reads as an
    # intermittent fault rather than an empty database.
    if len(names) == 0 or known_embeddings.shape[0] == 0:
        return "Unknown", 0.0

    normalized_embedding = embedding / np.linalg.norm(embedding)
    scores = known_embeddings @ normalized_embedding
    best_index = int(np.argmax(scores))
    score = float(scores[best_index])
    if score < MATCH_THRESHOLD:
        return "Unknown", score
    return names[best_index], score


def _write_pid_file() -> None:
    """Record this process id so the dashboard's restart button can stop us.

    The restart endpoint used to shell out to `pkill`, which does not exist on
    Windows. A pid file works identically on both platforms.
    """
    try:
        sentra_paths.ensure_data_dirs()
        sentra_paths.ENGINE_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        # Not fatal: the engine still detects faces, only the in-app restart
        # button loses its handle on this process.
        print(f"Warning: could not write pid file: {exc}")


def _make_recognition_worker(
    cam: CameraRuntime,
    model: FaceAnalysis,
    model_lock: threading.Lock,
    enrolled: EnrolledRegistry,
    running: threading.Event,
    visitors: VisitorRegistry | None = None,
):
    """Build the recognition_worker function for one AI-enabled camera.

    InsightFace's model is shared across every AI camera (loading it once is
    ~600MB and several seconds; N cameras should not pay that N times), guarded
    by `model_lock` since concurrent calls into the same onnxruntime session
    from multiple threads are not something to rely on being safe. Each
    camera's FightDetector is NOT shared — see anomaly_detection.py.
    """

    def recognition_worker() -> None:
        ai_width, ai_height = AI_FRAME_SIZE
        cycle = 0

        while running.is_set():
            try:
                # Read once per frame rather than closing over the roster: this
                # is what lets a newly registered person or an imported data
                # pack take effect without restarting the engine.
                names, known_embeddings = enrolled.get()

                frame = cam.stream.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_height, frame_width = frame.shape[:2]
                small_frame = cv2.resize(frame, (ai_width, ai_height))
                with model_lock:
                    detected_faces = model.get(small_frame)
                current_results: list[tuple[int, int, int, int, str, float, str]] = []
                overstay_alerts: list[dict] = []
                # Face boxes in small_frame coordinates, for matching against
                # the pose tracker which runs on that same small_frame.
                faces_for_anomaly: list[tuple[list[int], str]] = []

                x_scale = frame_width / ai_width
                y_scale = frame_height / ai_height

                print(f"[{cam.name}] Faces detected:", len(detected_faces))

                for face in detected_faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)

                    label, score, visitor = identify(
                        face.normed_embedding,
                        names,
                        known_embeddings,
                        visitors,
                    )

                    kind = "unknown"
                    if visitor is not None:
                        overdue = visitor_is_overdue(visitor)
                        kind = "visitor_overdue" if overdue else "visitor"
                        print(
                            f"[{cam.name}] {label} ({score:.2f}) "
                            f"[visitor{' — PASS EXPIRED' if overdue else ''}]"
                        )
                        if overdue:
                            # Rate-limited to one alert per visitor per hour
                            # inside visitor_store; None means suppressed.
                            alert = visitor_store.log_overstay(
                                visitor["id"], visitor["name"], cam.name,
                                visitor["expires_at"],
                            )
                            if alert is not None:
                                print(
                                    f"[{cam.name}] VISITOR OVERSTAY: {visitor['name']} "
                                    f"(pass expired {visitor['expires_at']})"
                                )
                                overstay_alerts.append(alert)
                    elif label != "Unknown":
                        kind = "enrolled"
                        print(f"[{cam.name}] {label} ({score:.2f})")
                    else:
                        print(f"[{cam.name}] {label} ({score:.2f})")

                    # Log recognized people only — visitors included, so the
                    # activity feed shows where a visitor has been, same as
                    # anyone else.
                    if label != "Unknown":
                        log_detection(label, cam.name)

                    faces_for_anomaly.append(([int(x1), int(y1), int(x2), int(y2)], label))

                    # Always add the detection so boxes are drawn,
                    # even if the face is Unknown.
                    current_results.append(
                        (
                            int(x1 * x_scale),
                            int(y1 * y_scale),
                            int(x2 * x_scale),
                            int(y2 * y_scale),
                            label,
                            score,
                            kind,
                        )
                    )

                with cam.results_lock:
                    cam.latest_results = current_results

                # Overstays reach the dashboard the same way fights do, so the
                # real-time banner fires on the event itself rather than
                # waiting up to 5s for the next poll.
                cam.publish_anomalies(
                    [
                        {
                            "type": "visitor_overstay",
                            "confidence": 1.0,
                            "persons_involved": [alert["visitor_name"]],
                            "bounding_boxes": [],
                            "timestamp": alert["detected_at"],
                            "visitor_id": alert["visitor_id"],
                            "expires_at": alert["expires_at"],
                            "overstay_seconds": alert["overstay_seconds"],
                        }
                        for alert in overstay_alerts
                    ]
                )

                if cam.detector is not None and cycle % ANOMALY_EVERY_N_CYCLES == 0:
                    try:
                        anomalies, _people = cam.detector.analyze_frame(
                            small_frame, faces_for_anomaly
                        )
                    except Exception as exc:
                        print(f"[{cam.name}] Anomaly detection error:", exc)
                        anomalies = []

                    for anomaly in anomalies:
                        # Scale the boxes from AI-frame to full-frame coords so
                        # they line up with the face boxes everywhere else.
                        anomaly["bounding_boxes"] = [
                            [
                                int(box[0] * x_scale),
                                int(box[1] * y_scale),
                                int(box[2] * x_scale),
                                int(box[3] * y_scale),
                            ]
                            for box in anomaly["bounding_boxes"]
                        ]

                        print(
                            f"[{cam.name}] ANOMALY: {anomaly['type']} "
                            f"({anomaly['confidence']:.2f}) "
                            f"{' vs '.join(anomaly['persons_involved'])}"
                        )
                        log_anomaly(
                            anomaly["type"],
                            anomaly["confidence"],
                            anomaly["persons_involved"],
                            anomaly["bounding_boxes"],
                            cam.name,
                        )

                    cam.publish_anomalies(anomalies)

                cycle += 1
                time.sleep(AI_INTERVAL_SECONDS)

            except Exception as e:
                print(f"[{cam.name}] Recognition worker error:", e)
                time.sleep(1)

    return recognition_worker


def _make_stream_worker(cam: CameraRuntime, ws_url: str, running: threading.Event):
    """Build the stream_worker function pushing one camera to one dashboard URL.

    Best-effort and independent of every other camera's stream_worker and of
    that camera's own recognition_worker: a dashboard being down, or this
    camera having no AI enabled, never blocks anything else.
    """

    def stream_worker() -> None:
        ws_conn = None
        # Tracks which anomaly events this (camera, backend) pair has already
        # sent, so a fight raises exactly one alert per dashboard rather than
        # one per frame for as long as the people stay in view.
        last_anomaly_event_id = -1
        while running.is_set():
            try:
                if ws_conn is None:
                    ws_conn = websocket.create_connection(ws_url, timeout=2)
                    print(f"[{cam.name}] Connected to dashboard backend for live streaming: {ws_url}")

                frame = cam.stream.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                frame_height, frame_width = frame.shape[:2]
                if frame_width > STREAM_MAX_WIDTH:
                    scale = STREAM_MAX_WIDTH / frame_width
                    out_frame = cv2.resize(
                        frame, (int(frame_width * scale), int(frame_height * scale))
                    )
                else:
                    scale = 1.0
                    out_frame = frame

                with cam.results_lock:
                    results_to_send = cam.latest_results.copy()

                faces_payload = [
                    {
                        "name": label,
                        "box": [
                            int(x1 * scale),
                            int(y1 * scale),
                            int(x2 * scale),
                            int(y2 * scale),
                        ],
                        "score": round(score, 3),
                        # enrolled | visitor | visitor_overdue | unknown —
                        # so the dashboard can label a temporary pass as one
                        # rather than showing a visitor as permanent staff.
                        "kind": kind,
                    }
                    for x1, y1, x2, y2, label, score, kind in results_to_send
                ]

                with cam.anomalies_lock:
                    new_anomalies = [
                        anomaly
                        for anomaly in cam.latest_anomalies
                        if anomaly["event_id"] > last_anomaly_event_id
                    ]
                if new_anomalies:
                    last_anomaly_event_id = new_anomalies[-1]["event_id"]

                anomalies_payload = [
                    {
                        "type": anomaly["type"],
                        "confidence": anomaly["confidence"],
                        "persons_involved": anomaly["persons_involved"],
                        "boxes": [
                            [
                                int(box[0] * scale),
                                int(box[1] * scale),
                                int(box[2] * scale),
                                int(box[3] * scale),
                            ]
                            for box in anomaly["bounding_boxes"]
                        ],
                        "timestamp": anomaly["timestamp"],
                        # Only a visitor overstay carries these; a fight has no
                        # equivalent, and the dashboard reads them per type.
                        "visitor_id": anomaly.get("visitor_id"),
                        "expires_at": anomaly.get("expires_at"),
                        "overstay_seconds": anomaly.get("overstay_seconds"),
                    }
                    for anomaly in new_anomalies
                ]

                ok, encoded = cv2.imencode(
                    ".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
                )
                if not ok:
                    continue

                payload = json.dumps(
                    {
                        "camera_id": cam.id,
                        "camera_name": cam.name,
                        "frame": base64.b64encode(encoded.tobytes()).decode("ascii"),
                        "frame_width": out_frame.shape[1],
                        "frame_height": out_frame.shape[0],
                        "faces": faces_payload,
                        "anomalies": anomalies_payload,
                    }
                )
                ws_conn.send(payload)
                time.sleep(STREAM_INTERVAL_SECONDS)

            except Exception as e:
                if ws_conn is not None:
                    print(f"[{cam.name}] Dashboard stream disconnected ({ws_url}):", e)
                try:
                    if ws_conn is not None:
                        ws_conn.close()
                except Exception:
                    pass
                ws_conn = None
                time.sleep(2)

    return stream_worker


def _draw_overlay(frame: np.ndarray, cam: CameraRuntime, running: threading.Event) -> None:
    """Draw face boxes + a small status line onto `frame` in place, for the
    local debug preview window. Not sent anywhere — the dashboard gets its own
    unannotated-but-boxed frame from stream_worker."""
    with cam.results_lock:
        results_to_draw = cam.latest_results.copy()

    for x1, y1, x2, y2, label, score, kind in results_to_draw:
        # BGR. An overdue visitor is the one case that must not read as "fine
        # at a glance", so it gets red rather than the Unknown amber.
        color = {
            "enrolled": (0, 200, 0),
            "visitor": (230, 160, 0),
            "visitor_overdue": (0, 0, 255),
        }.get(kind, (0, 165, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        name_text = label.upper()
        if kind == "visitor":
            name_text += " (VISITOR)"
        elif kind == "visitor_overdue":
            name_text += " (PASS EXPIRED)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        name_scale = 0.85
        name_size, _ = cv2.getTextSize(name_text, font, name_scale, 2)
        label_width = name_size[0] + 20
        label_height = name_size[1] + 16

        label_top = y1 - label_height - 8
        if label_top < 0:
            label_top = min(frame.shape[0] - label_height, y2 + 8)
        label_left = max(0, min(x1, frame.shape[1] - label_width))
        label_bottom = label_top + label_height

        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_left + label_width, label_bottom),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            name_text,
            (label_left + 10, label_top + name_size[1] + 7),
            font,
            name_scale,
            (0, 0, 0),
            2,
        )

    cv2.putText(
        frame,
        f"{cam.name} | AI: {'on' if cam.ai_enabled else 'off'} | faces: {len(results_to_draw)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def main() -> int:
    _write_pid_file()
    enrolled = EnrolledRegistry()
    enrolled.reload_if_changed()
    _startup_names, _ = enrolled.get()
    print(
        f"Loaded {len(_startup_names)} enrolled person(s)"
        f"{': ' + ', '.join(_startup_names) if _startup_names else ''}"
    )
    model = load_model()
    model_lock = threading.Lock()

    running = threading.Event()
    running.set()

    # Started before the cameras open, for the same reason the pose model is:
    # if visitor recognition is broken, that must be visible in the log rather
    # than masked by a camera that failed first.
    visitors = VisitorRegistry()
    visitors.start(running)
    if visitors.last_error:
        print(f"Temporary Pass unavailable (visitor database error): {visitors.last_error}")

    # Watches the enrolled database for changes from this point on, so someone
    # registered in the dashboard — or a whole roster imported as a data pack —
    # is recognised within seconds rather than after an engine restart.
    enrolled.start(running)

    camera_configs = camera_store.load_cameras()
    print(f"Configured camera(s): {len(camera_configs)}")

    cameras: list[CameraRuntime] = []
    for config in camera_configs:
        cam = CameraRuntime(config)

        # Loaded before the camera stream is opened, same ordering rationale as
        # before this file supported multiple cameras: if the pose model can't
        # load, face recognition on this camera must carry on regardless, but
        # that graceful degradation makes the failure invisible unless it's
        # reported before anything else (like a dead camera) can mask it.
        if cam.ai_enabled and ANOMALY_DETECTION_ENABLED:
            try:
                detector = anomaly_detection.FightDetector(label=cam.name)
                detector.load_pose_model()
                cam.detector = detector
            except Exception as exc:
                print(f"[{cam.name}] Fight detection disabled (could not load pose model): {exc}")

        try:
            cam.stream = LatestFrameStream(cam.rtsp_url)
            cam.stream.start()
            cam.online = True
            print(f"[{cam.name}] Camera connected ({cam.rtsp_url}).")
        except RuntimeError as exc:
            # Deliberately does not raise: one unreachable camera must not take
            # the others down with it. Formal_Code/camera_onvif_access notes
            # explain why cameras go unreachable (DHCP IP changes, power).
            print(f"[{cam.name}] Could not open the RTSP camera stream: {exc}")
            cam.online = False

        cameras.append(cam)

    online_cameras = [c for c in cameras if c.online]
    if not online_cameras:
        # Preserves the exact substring ("Could not open the RTSP") that
        # Start Sentra.command greps for when reporting camera failures.
        print("Could not open the RTSP camera stream for any configured camera.")
        return 1

    workers: list[threading.Thread] = []
    for cam in online_cameras:
        if cam.ai_enabled:
            worker = threading.Thread(
                target=_make_recognition_worker(
                    cam, model, model_lock, enrolled, running, visitors
                ),
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        for ws_url in DASHBOARD_WS_URLS:
            streamer = threading.Thread(
                target=_make_stream_worker(cam, ws_url, running), daemon=True
            )
            streamer.start()
            workers.append(streamer)

    if SHOW_PREVIEW:
        print(
            f"Running with {len(online_cameras)}/{len(cameras)} camera(s) online. "
            "Press Q in any preview window to quit."
        )
    else:
        print(
            f"Running with {len(online_cameras)}/{len(cameras)} camera(s) online. "
            "Preview windows are off; watch the Live Monitor tab in the dashboard."
        )

    displayed_frames = 0
    fps_started_at = time.perf_counter()
    # Set once, if highgui turns out to be unusable at runtime. Recognition and
    # the dashboard stream do not depend on it, so a preview failure downgrades
    # to headless rather than stopping the engine.
    preview_ok = SHOW_PREVIEW

    try:
        while True:
            any_frame = False
            for cam in online_cameras:
                frame = cam.stream.read()
                if frame is None:
                    continue
                any_frame = True
                # Overlay drawing is for the preview only — the dashboard gets
                # raw frames plus box coordinates and draws its own. Skipping it
                # headless saves the work rather than doing it for nobody.
                if preview_ok:
                    _draw_overlay(frame, cam, running)
                    try:
                        cv2.imshow(f"Sentra — {cam.name}", frame)
                    except cv2.error as exc:
                        print(f"Preview windows unavailable, continuing headless: {exc}")
                        preview_ok = False

            if any_frame:
                displayed_frames += 1
                elapsed = time.perf_counter() - fps_started_at
                # Printed via imshow's own window title update is unnecessary
                # noise per-camera; overall FPS is logged instead, occasionally.
                if displayed_frames % 150 == 0 and elapsed:
                    print(f"Display FPS (avg, all cameras combined): {displayed_frames / elapsed:.1f}")

            if preview_ok:
                # Doubles as the loop's pacing: waitKey yields for ~1ms and
                # pumps the highgui event queue.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # No window to pump, so nothing is yielding the CPU. Without
                # this the loop spins at whatever rate stream.read() returns.
                time.sleep(HEADLESS_LOOP_SLEEP)
    finally:
        running.clear()
        for worker in workers:
            worker.join(timeout=1)
        for cam in cameras:
            if cam.stream is not None:
                cam.stream.stop()
        if SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
