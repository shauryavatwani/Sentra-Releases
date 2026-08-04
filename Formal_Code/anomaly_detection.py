"""Multi-person fight detection for the Sentra AI engine.

This module sits alongside the InsightFace recognition in face_recognition.py.
InsightFace answers "who is in frame"; this module answers "what is happening".

Pipeline
--------
    frame
      -> YOLOv8-pose  : detects EVERY person in the frame + 17 body keypoints
      -> ByteTrack    : gives each person a stable track_id across frames
      -> face mapping : links InsightFace names to those tracks by box containment
      -> history      : per-track motion history (torso centre, wrists, bbox)
      -> classifier   : scores each PAIR of people for fight-like behaviour

Why YOLOv8-pose and not MediaPipe Pose: MediaPipe Pose returns a single
skeleton per frame, so it cannot express "person A vs person B" at all. Fight
detection is inherently a pairwise question, so the pose stage has to be
multi-person or the rest of the logic has nothing to compare.

Everything here is scale-invariant: distances and speeds are divided by the
person's own bounding-box height, so someone far from the camera (small in
pixels) is measured the same way as someone close to it.

One FightDetector per camera
-----------------------------
All tracking state (motion history, pair scores, remembered identities) is
per-instance, not module-level, and — this is the part that actually forces
it — so is the YOLO model object. `model.track(persist=True)` keeps its
ByteTrack state *inside that model object*, invisible to and unmanaged by
any of the dicts below. Two cameras sharing one YOLO instance would silently
share tracker slots: track ids would collide, motion histories would blend,
and the classifier could report a "fight" between two people who are each on
a different physical camera. Multi-camera support is therefore one
FightDetector — and one YOLO model — per camera, not a shared pipeline fed
from multiple sources.

Module-level functions below (`detect_people`, `analyze_frame`, etc.) are a
thin facade over a lazily-created default FightDetector, kept only so
existing single-camera callers and tests/test_anomaly_detection.py don't
have to change. New code (the multi-camera engine) should construct its own
FightDetector per camera instead.
"""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

import sentra_paths

# --------------------------- Tunable thresholds ---------------------------
# These are the knobs worth touching when tuning against a real camera.
# Shared across every FightDetector instance — they are properties of the
# detection method, not of any one camera.

# Absolute path so the weights resolve to one fixed file regardless of the
# working directory the engine was launched from. Passing a bare filename makes
# ultralytics look in the cwd and re-download when it isn't there — which both
# scatters copies around and silently needs internet at startup.
# Resolved via sentra_paths because a packaged build ships the weights beside
# the exe rather than next to this source file.
MODEL_PATH = sentra_paths.pose_model_path()
PERSON_CONF = 0.4               # min YOLO confidence to accept a person
KEYPOINT_CONF = 0.3             # min confidence for a single keypoint to be usable

# Proximity: torso-to-torso distance measured in "body heights".
# Full score at or inside PROXIMITY_FULL (about arm's length), tapering to
# zero at PROXIMITY_LIMIT. Two people can't occupy the same space, so the
# ramp starts at a realistic fighting distance rather than at zero.
PROXIMITY_FULL = 0.6
PROXIMITY_LIMIT = 1.4

# Limb speed measured in body-heights per second, as path length rather than
# net displacement — a punch that lands and retracts covers a lot of distance
# but ends up back where it started, so net displacement would read as zero.
SPEED_REFERENCE = 1.2

# Per-segment movement below this (in body heights) is treated as zero when
# summing wrist path length. YOLO keypoint estimates wobble by a couple of
# pixels frame to frame even on a person standing perfectly still, and because
# path length *sums* every segment, that jitter accumulates into phantom speed
# — a motionless person reads as gently moving. Deadbanding each segment means
# the motion signal measures movement rather than noise, which is what makes
# MOTION_FLOOR below meaningful rather than arbitrary.
WRIST_JITTER_DEADBAND = 0.02

# Contact: how close a wrist has to get to the other person's torso,
# again in body heights.
CONTACT_LIMIT = 0.45

# Score weighting. These must sum to 1.0, and the rule that gives them their
# shape is:
#
#     PROXIMITY_WEIGHT + CONTACT_WEIGHT < FIGHT_THRESHOLD
#
# Proximity and contact are both *static distance* measurements taken from a
# single frame — torso-to-torso and wrist-to-torso — and they are strongly
# correlated: people who are close together also have their hands near each
# other. They are emphatically not independent evidence. If the two of them can
# clear the threshold between them, then a hug, a shoulder squeeze or two
# people talking at arm's length with their hands up scores as a fight while
# nobody moves at all. Keeping their sum below the threshold means a pair can
# never be flagged on posture alone; something has to actually be *happening*.
PROXIMITY_WEIGHT = 0.30
MOTION_WEIGHT = 0.50
CONTACT_WEIGHT = 0.20

# Score weighting and firing rules.
FIGHT_THRESHOLD = 0.55      # confidence needed to consider the pair fighting

# Motion is not merely weighted, it is *required*: a pair scoring below this on
# the motion component is never flagged, however close together they are and
# however much their hands overlap. The weights above already stop stillness
# from reaching the threshold on its own, but this states the intent directly
# rather than leaving it as an emergent property of three numbers that a later
# retune could quietly undo. A fight is a thing people do, not a way they stand.
#
# This is the knob to tune first if real fights are being missed — raise it to
# suppress more false alarms, lower it to catch subtler scuffles. Set
# SENTRA_ANOMALY_DEBUG=1 to log every scored pair with its components, which is
# how you find the right value from real footage instead of guessing.
MOTION_FLOOR = 0.45

CONSECUTIVE_HITS = 2        # must score above threshold this many evaluations in a row
PAIR_COOLDOWN_SECONDS = 15  # don't re-log the same pair more often than this
HISTORY_LENGTH = 8          # motion samples kept per track
STALE_TRACK_SECONDS = 3.0   # forget tracks not seen for this long

# Log every scored pair and its component breakdown, not just the ones that
# fire. Off by default because it prints per pair per evaluation; the point of
# it is that a false positive or a missed fight is otherwise invisible — the
# log only ever showed alerts that *did* fire, so there was no way to see how
# close a near-miss came or which component was responsible.
DEBUG_SCORES = os.environ.get("SENTRA_ANOMALY_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on",
}

# A track needs this many motion samples before it can trigger an alert. A
# person who just walked into frame has no meaningful velocity history yet, and
# scoring them immediately is how a brand-new track produces a bogus alert.
MIN_TRACK_SAMPLES = 3

# A pair is only scored if both people have at least this many confident torso
# keypoints. Below that the pose is too uncertain to reason about, and feeding
# garbage coordinates into the distance maths invents alerts out of noise.
MIN_TORSO_KEYPOINTS = 2

# COCO-17 keypoint indices produced by YOLOv8-pose.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
TORSO_POINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)


# --------------------------- Pure helper functions --------------------------
# These touch no per-camera state, so they stay as plain module-level
# functions rather than methods, and are shared by every FightDetector.


def _containment(face_box, person_box) -> float:
    """Fraction of the face box that lies inside the person box.

    Deliberately NOT IoU: a face box is a small fraction of a whole-body box,
    so even a perfect match scores near-zero on IoU. What we actually want to
    know is "is this face inside this body", which is containment.
    """
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box

    ix1, iy1 = max(fx1, px1), max(fy1, py1)
    ix2, iy2 = min(fx2, px2), min(fy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    face_area = (fx2 - fx1) * (fy2 - fy1)
    return float(intersection / face_area) if face_area > 0 else 0.0


def _mean_point(person, indices) -> np.ndarray | None:
    """Average of the confident keypoints among `indices`, or None."""
    points = [
        person["keypoints"][i]
        for i in indices
        if person["kp_conf"][i] >= KEYPOINT_CONF
    ]
    if not points:
        return None
    return np.mean(points, axis=0)


def _confident_torso_count(person) -> int:
    """How many of the four torso keypoints are trustworthy this frame."""
    return int(sum(person["kp_conf"][i] >= KEYPOINT_CONF for i in TORSO_POINTS))


def _body_height(person) -> float:
    """Person's bbox height — the scale everything else is normalized by."""
    _, y1, _, y2 = person["box"]
    return max(float(y2 - y1), 1.0)


def _torso_centre(person) -> np.ndarray:
    """Centre of the torso, falling back to the bbox centre."""
    centre = _mean_point(person, TORSO_POINTS)
    if centre is not None:
        return centre
    x1, y1, x2, y2 = person["box"]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _contact_score(person_a, person_b, scale: float) -> float:
    """How close either person's wrists come to the other's torso."""
    closest = float("inf")

    for source, target in ((person_a, person_b), (person_b, person_a)):
        torso = _mean_point(target, TORSO_POINTS)
        if torso is None:
            continue
        for index in (LEFT_WRIST, RIGHT_WRIST):
            if source["kp_conf"][index] < KEYPOINT_CONF:
                continue
            wrist = source["keypoints"][index]
            closest = min(closest, float(np.hypot(wrist[0] - torso[0], wrist[1] - torso[1])))

    if closest == float("inf"):
        return 0.0
    return _clamp01((CONTACT_LIMIT - closest / scale) / CONTACT_LIMIT)


# ------------------------------- FightDetector ------------------------------


class FightDetector:
    """Stateful fight-detection pipeline for exactly one camera.

    Construct one per camera. Do not share an instance across cameras and do
    not share its `_pose_model` — see the module docstring for why.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label  # for log messages only, e.g. a camera name
        self._pose_model = None

        # Module-level history in the pre-refactor version of this file, now
        # per-instance. Keyed by ByteTrack id, which is only unique within
        # this instance's own model/tracker.
        self._track_history: dict[int, deque] = {}
        self._pair_hit_counts: dict[tuple[int, int], int] = {}
        self._pair_last_logged: dict[tuple[int, int], float] = {}

        # Remembered identity per track: {track_id: {name: times_seen}}.
        # InsightFace only names a person in frames where their face is
        # actually detectable, and during a fight it very often isn't — heads
        # turn away, motion blurs, bodies occlude each other. ByteTrack keeps
        # the person's id stable through all of that, so once a face has been
        # matched to a track we keep using that name for as long as the track
        # lives. Votes rather than a single assignment, so one bad frame can't
        # permanently mislabel someone. This deliberately does not touch
        # InsightFace's matching or threshold — it only remembers what
        # InsightFace already decided.
        self._track_identity: dict[int, dict[str, int]] = {}

    # -- Model loading --------------------------------------------------

    def load_pose_model(self):
        """Load this instance's own YOLOv8-pose model, lazily.

        Deliberately not shared with any other FightDetector: `model.track`
        keeps its ByteTrack state on the model object itself, so a shared
        model would mean shared (and colliding) track ids across cameras.
        Kept lazy so importing this module (e.g. from a test) doesn't pay the
        model-loading cost or require the weights to be present.
        """
        if self._pose_model is None:
            from ultralytics import YOLO  # imported here so the module imports cheaply

            tag = f" [{self.label}]" if self.label else ""
            print(f"Loading YOLOv8-pose model for anomaly detection{tag} ({MODEL_PATH})...")
            self._pose_model = YOLO(str(MODEL_PATH))
            print(f"Pose model loaded{tag}.")
        return self._pose_model

    # -- Lifecycle --------------------------------------------------------

    def reset_state(self) -> None:
        """Clear all tracking state (used by tests and on engine restart)."""
        self._track_history.clear()
        self._pair_hit_counts.clear()
        self._pair_last_logged.clear()
        self._track_identity.clear()

    # -- Detection ----------------------------------------------------------

    def detect_people(self, frame) -> list[dict]:
        """Run YOLOv8-pose + ByteTrack and return one dict per tracked person.

        Each person is {track_id, box (x1,y1,x2,y2), keypoints (17,2), kp_conf (17,)}.
        Detections that ByteTrack could not assign an id to are skipped, because
        without a stable id there is no way to measure motion over time.
        """
        model = self.load_pose_model()
        results = model.track(
            frame,
            persist=True,          # keep ByteTrack ids alive between calls
            tracker="bytetrack.yaml",
            classes=[0],           # person class only
            conf=PERSON_CONF,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            return []

        track_ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None and keypoints.xy is not None:
            kp_xy = keypoints.xy.cpu().numpy()
            kp_conf = (
                keypoints.conf.cpu().numpy()
                if keypoints.conf is not None
                else np.ones(kp_xy.shape[:2], dtype=np.float32)
            )
        else:
            kp_xy = np.zeros((len(track_ids), 17, 2), dtype=np.float32)
            kp_conf = np.zeros((len(track_ids), 17), dtype=np.float32)

        people = []
        for i, track_id in enumerate(track_ids):
            people.append(
                {
                    "track_id": int(track_id),
                    "box": xyxy[i].tolist(),
                    "keypoints": kp_xy[i],
                    "kp_conf": kp_conf[i],
                }
            )
        return people

    # -- Face <-> track linking ----------------------------------------------

    def associate_faces_to_tracks(
        self, faces, people, containment_threshold: float = 0.5
    ) -> dict[int, str]:
        """Map each tracked person to a recognized name, where one is available.

        `faces` are InsightFace results (each with .bbox and a resolved label),
        passed in as (box, name) pairs. Tracks with no matching face stay
        "Unknown" — an unidentified person in a fight is still worth alerting on.

        When a face falls inside more than one person box (overlapping people),
        the nose keypoint breaks the tie: the body whose head is closest to the
        face wins.
        """
        if not people:
            return {}

        mapping = {person["track_id"]: "Unknown" for person in people}

        for face_box, name in faces:
            best_person = None
            best_score = containment_threshold
            best_head_distance = float("inf")

            face_cx = (face_box[0] + face_box[2]) / 2
            face_cy = (face_box[1] + face_box[3]) / 2

            for person in people:
                score = _containment(face_box, person["box"])
                if score < containment_threshold:
                    continue

                # Tie-break on head proximity when the face sits inside several boxes.
                head_distance = float("inf")
                if person["kp_conf"][NOSE] >= KEYPOINT_CONF:
                    nose = person["keypoints"][NOSE]
                    head_distance = float(np.hypot(nose[0] - face_cx, nose[1] - face_cy))

                better = score > best_score or (
                    abs(score - best_score) < 1e-6 and head_distance < best_head_distance
                )
                if better:
                    best_person = person
                    best_score = score
                    best_head_distance = head_distance

            if best_person is not None:
                mapping[best_person["track_id"]] = name

        # Fold this frame's matches into the remembered identity, then answer from
        # memory. A track keeps its name through frames where the face isn't
        # visible, which is most frames of an actual fight.
        for track_id, name in mapping.items():
            if name != "Unknown":
                votes = self._track_identity.setdefault(track_id, {})
                votes[name] = votes.get(name, 0) + 1

        return {
            track_id: self.resolved_identity(track_id)
            for track_id in mapping
        }

    def resolved_identity(self, track_id: int) -> str:
        """Best-known name for a track, or "Unknown" if it was never recognized."""
        votes = self._track_identity.get(track_id)
        if not votes:
            return "Unknown"
        return max(votes, key=votes.get)

    # -- Motion history -------------------------------------------------

    def _track_is_mature(self, track_id: int) -> bool:
        """True once a track has enough motion history to be judged."""
        history = self._track_history.get(track_id)
        return history is not None and len(history) >= MIN_TRACK_SAMPLES

    def update_track_history(self, people, now: float | None = None) -> None:
        """Record the current motion sample for every tracked person."""
        if now is None:
            now = time.time()

        for person in people:
            track_id = person["track_id"]
            history = self._track_history.setdefault(track_id, deque(maxlen=HISTORY_LENGTH))

            wrists = {}
            for side, index in (("left", LEFT_WRIST), ("right", RIGHT_WRIST)):
                if person["kp_conf"][index] >= KEYPOINT_CONF:
                    wrists[side] = np.array(person["keypoints"][index], dtype=float)

            history.append(
                {
                    "time": now,
                    "torso": _torso_centre(person),
                    "wrists": wrists,
                    "height": _body_height(person),
                }
            )

        # Forget people who have left the frame so the dicts don't grow forever.
        stale = [
            track_id
            for track_id, history in self._track_history.items()
            if history and now - history[-1]["time"] > STALE_TRACK_SECONDS
        ]
        for track_id in stale:
            self._track_history.pop(track_id, None)
            # Drop the remembered name too: ByteTrack reuses ids over time, so
            # holding one would eventually paste an old name onto a new person.
            self._track_identity.pop(track_id, None)
            for pair in [p for p in self._pair_hit_counts if track_id in p]:
                self._pair_hit_counts.pop(pair, None)

    def limb_speed(self, track_id: int) -> float:
        """Fastest wrist speed for this person, in body-heights per second.

        Wrist speed rather than whole-body speed is what separates a fight from
        someone simply walking quickly: punching moves the hands fast while the
        body itself stays roughly in place.

        Measured as total path length over the history window, not start-to-end
        displacement. A punch thrown and retracted returns the wrist to nearly its
        original position, so displacement would score it as motionless.

        Segments shorter than WRIST_JITTER_DEADBAND are discarded rather than
        summed. Path length is the right measure for punches but it is also
        pathologically good at accumulating noise: every frame contributes its
        keypoint wobble with a positive sign, so a perfectly still person builds
        up a steady phantom speed that grows with the window length. Dropping
        sub-deadband segments keeps that from being mistaken for movement.
        """
        history = self._track_history.get(track_id)
        if history is None or len(history) < 2:
            return 0.0

        elapsed = history[-1]["time"] - history[0]["time"]
        if elapsed <= 0:
            return 0.0

        height = history[-1]["height"]
        fastest = 0.0
        for side in ("left", "right"):
            travelled = 0.0
            for earlier, later in zip(history, list(history)[1:]):
                if side in earlier["wrists"] and side in later["wrists"]:
                    step = float(
                        np.linalg.norm(later["wrists"][side] - earlier["wrists"][side])
                    ) / height
                    if step >= WRIST_JITTER_DEADBAND:
                        travelled += step
            fastest = max(fastest, travelled / elapsed)
        return fastest

    # -- Fight classification ---------------------------------------------

    def score_pair(self, person_a, person_b) -> tuple[float, dict]:
        """Score one pair of people for fight-like behaviour.

        Returns (confidence, components). Three signals are combined:
          proximity - are they close enough to actually be fighting
          motion    - are their limbs moving violently fast
          contact   - are hands reaching into the other person's torso space

        All three matter: two people close together but still are talking; someone
        moving fast alone is running; contact without speed is a handshake.

        Note that the confidence returned here is not the whole decision.
        Proximity and contact are static, single-frame, mutually correlated
        measurements, so classify_fights additionally requires motion to clear
        MOTION_FLOOR before a pair can be flagged at all — see the weights
        block at the top of this module.
        """
        scale = (_body_height(person_a) + _body_height(person_b)) / 2

        centre_a, centre_b = _torso_centre(person_a), _torso_centre(person_b)
        distance = float(np.linalg.norm(centre_a - centre_b)) / scale
        proximity = _clamp01(
            (PROXIMITY_LIMIT - distance) / (PROXIMITY_LIMIT - PROXIMITY_FULL)
        )

        speed_a = self.limb_speed(person_a["track_id"])
        speed_b = self.limb_speed(person_b["track_id"])
        fast, slow = max(speed_a, speed_b), min(speed_a, speed_b)
        # Weighted toward the faster person so a one-sided assault still registers,
        # but the slower person still contributes — mutual motion is more fight-like.
        motion = _clamp01((0.7 * fast + 0.3 * slow) / SPEED_REFERENCE)

        contact = _contact_score(person_a, person_b, scale)

        confidence = (
            PROXIMITY_WEIGHT * proximity
            + MOTION_WEIGHT * motion
            + CONTACT_WEIGHT * contact
        )
        components = {
            "distance_body_heights": round(distance, 3),
            "proximity": round(proximity, 3),
            "motion": round(motion, 3),
            "contact": round(contact, 3),
            "speed_a": round(speed_a, 3),
            "speed_b": round(speed_b, 3),
        }
        return confidence, components

    def _name_for(self, track_id: int, face_map) -> str:
        """Remembered identity, falling back to whatever this frame supplied.

        The fallback matters when classify_fights is driven directly rather than
        through analyze_frame — without it, a caller's own face_map would be
        silently ignored.
        """
        remembered = self.resolved_identity(track_id)
        if remembered != "Unknown":
            return remembered
        return (face_map or {}).get(track_id, "Unknown")

    def classify_fights(self, people, face_map, now: float | None = None) -> list[dict]:
        """Check every pair of tracked people and return confirmed fight anomalies.

        A pair has to clear MOTION_FLOOR, then score above threshold on
        CONSECUTIVE_HITS evaluations in a row, before it fires. The motion
        requirement is what keeps a hug or a shoulder squeeze from scoring as a
        fight on closeness alone; the temporal requirement is what keeps a
        high-five or one noisy frame of bad keypoints from raising an alert.
        """
        if now is None:
            now = time.time()

        anomalies = []

        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                person_a, person_b = people[i], people[j]
                pair = tuple(sorted((person_a["track_id"], person_b["track_id"])))

                # Quality gates before any scoring: an uncertain pose or a track
                # that only just appeared cannot produce a trustworthy score, and
                # scoring it anyway is how noise turns into a false alert.
                if (
                    _confident_torso_count(person_a) < MIN_TORSO_KEYPOINTS
                    or _confident_torso_count(person_b) < MIN_TORSO_KEYPOINTS
                    or not self._track_is_mature(person_a["track_id"])
                    or not self._track_is_mature(person_b["track_id"])
                ):
                    self._pair_hit_counts.pop(pair, None)
                    continue

                confidence, components = self.score_pair(person_a, person_b)

                if DEBUG_SCORES:
                    tag = f"[{self.label}] " if self.label else ""
                    print(
                        f"{tag}PAIR {pair} conf={confidence:.3f} "
                        f"(threshold {FIGHT_THRESHOLD}) "
                        f"motion={components['motion']:.3f} "
                        f"(floor {MOTION_FLOOR}) "
                        f"proximity={components['proximity']:.3f} "
                        f"contact={components['contact']:.3f} "
                        f"speeds={components['speed_a']:.3f}/{components['speed_b']:.3f}",
                        flush=True,
                    )

                # Motion is a requirement, not just a contributor. Without this,
                # proximity and contact — two correlated static measurements —
                # decide the outcome between them, and standing close with hands
                # raised reads identically to fighting.
                if components["motion"] < MOTION_FLOOR:
                    self._pair_hit_counts.pop(pair, None)
                    continue

                if confidence < FIGHT_THRESHOLD:
                    self._pair_hit_counts.pop(pair, None)
                    continue

                hits = self._pair_hit_counts.get(pair, 0) + 1
                self._pair_hit_counts[pair] = hits
                if hits < CONSECUTIVE_HITS:
                    continue

                # Absent means "never fired" — distinct from "fired at t=0", which
                # a 0.0 default would wrongly suppress.
                last_logged = self._pair_last_logged.get(pair)
                if last_logged is not None and now - last_logged < PAIR_COOLDOWN_SECONDS:
                    continue
                self._pair_last_logged[pair] = now

                anomalies.append(
                    {
                        "type": "fight",
                        "confidence": round(float(confidence), 3),
                        # Remembered identity, not just this frame's face matches —
                        # faces are usually not detectable mid-fight.
                        "persons_involved": [
                            self._name_for(person_a["track_id"], face_map),
                            self._name_for(person_b["track_id"], face_map),
                        ],
                        "bounding_boxes": [
                            [int(v) for v in person_a["box"]],
                            [int(v) for v in person_b["box"]],
                        ],
                        "details": components,
                    }
                )

        return anomalies

    def analyze_frame(self, frame, faces, now: float | None = None) -> tuple[list[dict], list[dict]]:
        """Run the whole pipeline on one frame.

        `faces` is a list of (box, name) pairs from InsightFace, in the same
        coordinate space as `frame`. Returns (anomalies, people) — `people` is
        returned too so the caller can draw the tracked boxes if it wants.
        """
        if now is None:
            now = time.time()

        people = self.detect_people(frame)
        if len(people) < 2:
            # Fights need at least two people; still record history so that motion
            # measurements are warm the moment a second person walks in.
            self.update_track_history(people, now)
            return [], people

        face_map = self.associate_faces_to_tracks(faces, people)
        self.update_track_history(people, now)
        anomalies = self.classify_fights(people, face_map, now)
        return anomalies, people


# ------------------------- Module-level facade ------------------------------
# Kept so existing single-camera callers and tests/test_anomaly_detection.py
# keep working unchanged. New (multi-camera) code should use FightDetector
# directly instead of this shared default instance.

_default_detector = FightDetector()


def load_pose_model():
    return _default_detector.load_pose_model()


def reset_state() -> None:
    _default_detector.reset_state()


def detect_people(frame) -> list[dict]:
    return _default_detector.detect_people(frame)


def associate_faces_to_tracks(faces, people, containment_threshold: float = 0.5) -> dict[int, str]:
    return _default_detector.associate_faces_to_tracks(faces, people, containment_threshold)


def resolved_identity(track_id: int) -> str:
    return _default_detector.resolved_identity(track_id)


def update_track_history(people, now: float | None = None) -> None:
    _default_detector.update_track_history(people, now)


def limb_speed(track_id: int) -> float:
    return _default_detector.limb_speed(track_id)


def score_pair(person_a, person_b) -> tuple[float, dict]:
    return _default_detector.score_pair(person_a, person_b)


def classify_fights(people, face_map, now: float | None = None) -> list[dict]:
    return _default_detector.classify_fights(people, face_map, now)


def analyze_frame(frame, faces, now: float | None = None) -> tuple[list[dict], list[dict]]:
    return _default_detector.analyze_frame(frame, faces, now)
