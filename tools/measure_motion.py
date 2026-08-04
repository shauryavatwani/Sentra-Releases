"""Measure what the fight detector's motion signal actually reads on real people.

Why this exists
---------------
MOTION_FLOOR, SPEED_REFERENCE and WRIST_JITTER_DEADBAND in anomaly_detection.py
were all chosen against *synthetic* keypoint sequences — an idealised reach
pattern with clean, evenly spaced samples. They have never been compared to a
real human on a real camera at the real sampling cadence. Twice now that gap has
produced a shipped bug: first a hug scoring as a fight, then (the reason this
tool exists) a real fight producing no alert at all.

This tool closes that gap. It runs the *production* pipeline — same frame size,
same anomaly cadence, same FightDetector — against a live camera or a video
file, and reports the distribution of the numbers the thresholds are compared
against, rather than whether an alert fired.

The key thing it exploits: ``limb_speed`` is computed **per track,
independently**. Calibrating the motion signal therefore does not need two
people fighting — one person throwing real punches produces the real
distribution of the exact quantity MOTION_FLOOR gates on. That turns a
"get two volunteers and stage a fight" blocker into a one-person, 30-second
measurement.

Usage
-----
    python3 tools/measure_motion.py --seconds 20 --label "punching the air"
    python3 tools/measure_motion.py --source clip.mp4 --label "real scuffle"

Run it once per behaviour (still / talking / gesturing / punching) and compare
the reported percentiles. The right MOTION_FLOOR is a value that sits in the gap
between the benign runs and the fight runs — read off measurements, not guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Formal_Code"))

import anomaly_detection as ad  # noqa: E402

# Mirror the production engine (face_recognition.py) exactly. Measuring at a
# different frame size or cadence would produce numbers that don't transfer:
# body-height normalisation depends on the former, and path-length motion
# depends on the latter.
AI_FRAME_SIZE = (480, 270)
AI_INTERVAL_SECONDS = 0.25
ANOMALY_EVERY_N_CYCLES = 2


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


class LatestFrame:
    """Keep only the newest frame from a live source, exactly as the engine does.

    This is not an optimisation, it is a correctness requirement for the
    measurement. RTSP delivers a continuous stream and the capture buffers it;
    reading sequentially while sleeping between reads returns progressively
    *older* frames, so wrist positions from several seconds ago get stamped with
    the current time. Limb speed is displacement over elapsed time, so that
    inflates or destroys every number this tool exists to report — the
    measurement would look plausible and be meaningless.

    face_recognition.py::LatestFrameStream solves this the same way (a reader
    thread plus CAP_PROP_BUFFERSIZE=1); mirroring it here is what makes these
    numbers transfer to production.
    """

    def __init__(self, source) -> None:
        self._capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG) \
            if isinstance(source, str) else cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def is_opened(self) -> bool:
        return self._capture.isOpened()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

    def read(self):
        with self._lock:
            return (self._frame is not None, None if self._frame is None else self._frame.copy())

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0",
                        help="camera index (default 0 = built-in webcam) or a video file path")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to measure for (live sources only)")
    parser.add_argument("--label", default="unlabelled",
                        help="what the person is doing, for the report")
    parser.add_argument("--out", default="",
                        help="optional path to write raw samples as JSON")
    parser.add_argument("--cadence", type=float, default=AI_INTERVAL_SECONDS * ANOMALY_EVERY_N_CYCLES,
                        help="seconds between anomaly evaluations (production default 0.50). "
                             "Lower it to test whether the sampling rate, rather than the "
                             "thresholds, is what's losing the motion signal: a punch lasting "
                             "~0.3s falls entirely between two samples taken 0.5s apart, so "
                             "path length collapses to the straight line between two unrelated "
                             "points in the punch cycle.")
    args = parser.parse_args()

    source: object = int(args.source) if args.source.isdigit() else args.source
    # A webcam index and an RTSP stream are both live: they never end on their
    # own, so the run is bounded by --seconds and paced to the production
    # cadence. A video file is neither — it ends by itself and must be read as
    # fast as possible, since sleeping between frames would just make a 20s clip
    # take minutes without changing a single measurement.
    is_live = isinstance(source, int) or str(source).startswith("rtsp://")

    capture = LatestFrame(source) if is_live else cv2.VideoCapture(source)
    if not (capture.is_opened() if is_live else capture.isOpened()):
        print(f"ERROR: could not open source {args.source!r}")
        return 1
    if is_live:
        capture.start()
        time.sleep(1.5)  # let the reader thread land a first frame

    detector = ad.FightDetector("measure")
    detector.load_pose_model()

    print(f"\nMeasuring: {args.label}")
    print(f"  source={args.source}  frame={AI_FRAME_SIZE}  "
          f"anomaly cadence={args.cadence:.2f}s")
    print(f"  thresholds in force: MOTION_FLOOR={ad.MOTION_FLOOR} "
          f"SPEED_REFERENCE={ad.SPEED_REFERENCE} "
          f"deadband={ad.WRIST_JITTER_DEADBAND}")
    if is_live:
        print(f"  recording for {args.seconds:.0f}s — go.\n")

    # Per-sample records of the quantity MOTION_FLOOR actually gates on.
    motion_samples: list[float] = []      # normalised, as compared to MOTION_FLOOR
    raw_speed_samples: list[float] = []   # body-heights/sec, before normalisation
    torso_conf_samples: list[int] = []    # confident torso keypoints per person
    people_counts: list[int] = []
    gated = 0
    scored = 0

    start = time.time()
    cycle = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            if is_live:
                time.sleep(0.05)
                continue
            break
        if is_live and time.time() - start > args.seconds:
            break

        small = cv2.resize(frame, AI_FRAME_SIZE)

        if True:
            now = time.time()
            people = detector.detect_people(small)
            detector.update_track_history(people, now)
            people_counts.append(len(people))

            for person in people:
                tid = person["track_id"]
                torso_conf_samples.append(ad._confident_torso_count(person))
                speed = detector.limb_speed(tid)
                if detector._track_is_mature(tid):
                    raw_speed_samples.append(speed)
                    # How this speed reads against MOTION_FLOOR when this person
                    # is the only one moving — the one-sided-assault weighting,
                    # which is the conservative case.
                    motion_samples.append(min(1.0, (0.7 * speed) / ad.SPEED_REFERENCE))

            # Pair-level gating, only meaningful with 2+ people in frame.
            if len(people) >= 2:
                for i in range(len(people)):
                    for j in range(i + 1, len(people)):
                        a, b = people[i], people[j]
                        if (ad._confident_torso_count(a) < ad.MIN_TORSO_KEYPOINTS
                                or ad._confident_torso_count(b) < ad.MIN_TORSO_KEYPOINTS
                                or not detector._track_is_mature(a["track_id"])
                                or not detector._track_is_mature(b["track_id"])):
                            gated += 1
                        else:
                            scored += 1

            elapsed = time.time() - start
            latest = motion_samples[-1] if motion_samples else 0.0
            print(f"  t={elapsed:5.1f}s people={len(people)} "
                  f"motion={latest:.3f} "
                  f"{'<-- would clear floor' if latest >= ad.MOTION_FLOOR else ''}",
                  flush=True)

        cycle += 1
        if is_live:
            time.sleep(args.cadence)

    capture.release()

    if not motion_samples:
        print("\nNo mature tracks were produced — nobody was detected long enough "
              "to measure. Check the camera actually sees a person.")
        return 1

    print(f"\n{'=' * 62}")
    print(f"RESULT — {args.label}")
    print(f"{'=' * 62}")
    print(f"  samples                {len(motion_samples)}")
    print(f"  people in frame        avg {np.mean(people_counts):.2f}, max {max(people_counts)}")
    print(f"  confident torso kps    avg {np.mean(torso_conf_samples):.2f} of 4 "
          f"(need {ad.MIN_TORSO_KEYPOINTS})")
    if gated or scored:
        print(f"  pair gating            {gated} gated / {gated + scored} evaluated")
    print()
    print(f"  raw wrist speed (body-heights/sec)")
    print(f"    median {percentile(raw_speed_samples, 50):.3f}   "
          f"p90 {percentile(raw_speed_samples, 90):.3f}   "
          f"max {max(raw_speed_samples):.3f}")
    print()
    print(f"  motion component (what MOTION_FLOOR={ad.MOTION_FLOOR} gates on)")
    print(f"    median {percentile(motion_samples, 50):.3f}   "
          f"p90 {percentile(motion_samples, 90):.3f}   "
          f"max {max(motion_samples):.3f}")
    cleared = sum(1 for m in motion_samples if m >= ad.MOTION_FLOOR)
    print(f"    cleared the floor in {cleared}/{len(motion_samples)} samples "
          f"({100 * cleared / len(motion_samples):.0f}%)")
    print()

    if args.out:
        Path(args.out).write_text(json.dumps({
            "label": args.label,
            "motion_samples": motion_samples,
            "raw_speed_samples": raw_speed_samples,
            "motion_floor": ad.MOTION_FLOOR,
            "speed_reference": ad.SPEED_REFERENCE,
            "deadband": ad.WRIST_JITTER_DEADBAND,
        }, indent=2), encoding="utf-8")
        print(f"  raw samples written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
