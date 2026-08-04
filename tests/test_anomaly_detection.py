"""Tests for anomaly_detection.py — no camera or model needed.

Mostly synthetic, but the final section replays *real* wrist-speed sequences
recorded from the production camera (tests/fixtures/measured_wrist_speeds.json).
That distinction matters: every synthetic test here passed while the detector
was shipping broken, twice — once alerting on hugs, once unable to report a real
fight at all. Synthetic cases verify the logic; only the measured ones verify
that the constants match a human being.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Formal_Code"))
import anomaly_detection as ad


def make_person(track_id, cx, cy, height=100.0, wrist_offset=(0, 0), conf=1.0):
    """Build a fake tracked person centred at (cx, cy)."""
    half_w, half_h = height * 0.25, height / 2
    kp = np.zeros((17, 2), dtype=float)
    kpc = np.full(17, conf, dtype=float)
    kp[ad.NOSE] = [cx, cy - half_h * 0.8]
    kp[ad.LEFT_SHOULDER] = [cx - half_w, cy - half_h * 0.3]
    kp[ad.RIGHT_SHOULDER] = [cx + half_w, cy - half_h * 0.3]
    kp[ad.LEFT_HIP] = [cx - half_w, cy + half_h * 0.3]
    kp[ad.RIGHT_HIP] = [cx + half_w, cy + half_h * 0.3]
    kp[ad.LEFT_WRIST] = [cx + wrist_offset[0], cy + wrist_offset[1]]
    kp[ad.RIGHT_WRIST] = [cx + wrist_offset[0], cy + wrist_offset[1]]
    return {
        "track_id": track_id,
        "box": [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        "keypoints": kp,
        "kp_conf": kpc,
    }


def run(label, frames, expect):
    """Feed frames (list of (people, t)) and report whether a fight fired."""
    ad.reset_state()
    fired = []
    for people, t in frames:
        ad.update_track_history(people, t)
        fired += ad.classify_fights(people, {p["track_id"]: "Unknown" for p in people}, t)
    ok = bool(fired) == expect
    conf = f" conf={fired[0]['confidence']}" if fired else ""
    print(f"{'PASS' if ok else 'FAIL'}  {label}: fired={bool(fired)}{conf}")
    return ok


results = []

# Punches thrown and retracted: the wrist ends up back where it started, which
# is exactly the case net displacement would miss and path length catches.
PUNCH_CYCLE = [0, 60, 5, 62, 3]

# 1. Two people close, trading punches -> should fire.
frames = []
for i, reach in enumerate(PUNCH_CYCLE):
    a = make_person(1, 100, 100, wrist_offset=(reach, 0))
    b = make_person(2, 170, 100, wrist_offset=(-reach, 0))
    frames.append(([a, b], i * 0.5))
results.append(run("fight: close + fast limbs + contact", frames, True))

# 2. Two people close but completely still (a conversation) -> must not fire.
frames = [([make_person(1, 100, 100), make_person(2, 170, 100)], i * 0.5) for i in range(4)]
results.append(run("conversation: close but static", frames, False))

# 3. Two people moving limbs fast but far apart -> must not fire.
frames = []
for i, reach in enumerate(PUNCH_CYCLE):
    a = make_person(1, 100, 100, wrist_offset=(reach, 0))
    b = make_person(2, 600, 100, wrist_offset=(-reach, 0))
    frames.append(([a, b], i * 0.5))
results.append(run("far apart: fast limbs, no proximity", frames, False))

# 4. Single person, fast movement -> no pair, must not fire.
frames = [([make_person(1, 100, 100, wrist_offset=(r, 0))], i * 0.5)
          for i, r in enumerate(PUNCH_CYCLE)]
results.append(run("single person alone", frames, False))

# 5. Temporal gate: one single high-scoring frame should not be enough.
a = make_person(1, 100, 100, wrist_offset=(60, 0))
b = make_person(2, 170, 100, wrist_offset=(-60, 0))
ad.reset_state()
ad.update_track_history([a, b], 0.0)
one = ad.classify_fights([a, b], {1: "Unknown", 2: "Unknown"}, 0.0)
ok = not one
print(f"{'PASS' if ok else 'FAIL'}  temporal gate: single frame does not fire")
results.append(ok)

# 6. Face-to-track association by containment (not IoU).
people = [make_person(1, 100, 100), make_person(2, 300, 100)]
# A face box sitting on person 1's head.
face_a = ([88, 55, 112, 80], "Shaurya Vatwani")
face_b = ([288, 55, 312, 80], "Ishan")
mapping = ad.associate_faces_to_tracks([face_a, face_b], people)
ok = mapping[1] == "Shaurya Vatwani" and mapping[2] == "Ishan"
print(f"{'PASS' if ok else 'FAIL'}  face->track association: {mapping}")
results.append(ok)

# 7. Unmatched face leaves the track as Unknown.
# Fresh state: with identity memory, a track that was named by an earlier
# check would legitimately keep that name, which is not what this asserts.
ad.reset_state()
mapping = ad.associate_faces_to_tracks([([900, 900, 920, 930], "Veer")], people)
ok = mapping[1] == "Unknown" and mapping[2] == "Unknown"
print(f"{'PASS' if ok else 'FAIL'}  unmatched face stays Unknown: {mapping}")
results.append(ok)

# 8. Containment beats IoU for a face inside a body box.
c = ad._containment([88, 55, 112, 80], people[0]["box"])
ok = c > 0.9
print(f"{'PASS' if ok else 'FAIL'}  containment of face in body = {c:.2f} (IoU would be ~0.05)")
results.append(ok)

# 9. Cooldown: one sustained fight must produce one alert, not one per frame.
ad.reset_state()
fired = []
for i in range(8):
    reach = PUNCH_CYCLE[i % len(PUNCH_CYCLE)]
    people_f = [make_person(1, 100, 100, wrist_offset=(reach, 0)),
                make_person(2, 170, 100, wrist_offset=(-reach, 0))]
    t = i * 0.5
    ad.update_track_history(people_f, t)
    fired += ad.classify_fights(people_f, {1: "A", 2: "B"}, t)
ok = len(fired) == 1
print(f"{'PASS' if ok else 'FAIL'}  cooldown: {len(fired)} alert(s) across 8 frames (expect 1)")
results.append(ok)

# 10. Stillness can never fire, however close two people stand.
#
# The previous version of this test checked only that each weight *individually*
# sat below the threshold — max(0.35, 0.40, 0.25) < 0.55 — and passed happily
# while the real bug was in the pair: proximity 0.35 + contact 0.25 = 0.60, over
# the 0.55 threshold with motion contributing exactly nothing. A hug scored as a
# fight and the test suite agreed it couldn't. Singles were never the risk;
# proximity and contact are both static distance measurements taken from one
# frame and they rise together, so it is specifically their *sum* that has to
# stay below the line.
ok = ad.PROXIMITY_WEIGHT + ad.CONTACT_WEIGHT < ad.FIGHT_THRESHOLD
print(f"{'PASS' if ok else 'FAIL'}  motionless pair cannot reach threshold: "
      f"proximity+contact={ad.PROXIMITY_WEIGHT + ad.CONTACT_WEIGHT:.2f} "
      f"< {ad.FIGHT_THRESHOLD}")
results.append(ok)

# Weights must still be a partition of 1.0, or "confidence" stops meaning
# anything comparable across releases and the threshold silently changes value.
total = ad.PROXIMITY_WEIGHT + ad.MOTION_WEIGHT + ad.CONTACT_WEIGHT
ok = abs(total - 1.0) < 1e-9
print(f"{'PASS' if ok else 'FAIL'}  weights sum to 1.0: {total}")
results.append(ok)

# ---------------------------------------------------------------------------
# False-positive regressions — the reported bug.
#
# Every scenario below is two people doing something ordinary at close range.
# All of them fired before the motion floor existed. These are the tests that
# would have caught the original bug, so they are the ones that matter most.
# ---------------------------------------------------------------------------

# A hug: as close as two people get, arms fully around the other's torso, but
# the arms are not moving fast. Maximum proximity and maximum contact — the
# exact combination that used to clear the threshold on its own.
frames = []
for i in range(5):
    # 40px apart at height 100 = 0.4 body heights: closer than "arm's length".
    a = make_person(1, 100, 100, wrist_offset=(35, 0))   # arm around B
    b = make_person(2, 140, 100, wrist_offset=(-35, 0))  # arm around A
    frames.append(([a, b], i * 0.5))
results.append(run("hug: maximum proximity + maximum contact, slow", frames, False))

# Two people standing at conversational distance, gesturing with their hands.
# Real hand movement, but nothing like fighting speed.
frames = []
GESTURE = [0, 12, 4, 14, 2]  # small back-and-forth, ~0.12 body heights
for i, reach in enumerate(GESTURE):
    a = make_person(1, 100, 100, wrist_offset=(reach, 0))
    b = make_person(2, 165, 100, wrist_offset=(-reach, 0))
    frames.append(([a, b], i * 0.5))
results.append(run("conversation: close, gesturing while talking", frames, False))

# A shoulder squeeze / pat on the back: one person's hand resting on the other,
# neither of them moving much.
frames = []
for i in range(5):
    a = make_person(1, 100, 100, wrist_offset=(50, -10))  # hand on B's shoulder
    b = make_person(2, 155, 100, wrist_offset=(0, 0))
    frames.append(([a, b], i * 0.5))
results.append(run("shoulder squeeze: contact without speed", frames, False))

# Keypoint jitter on two people standing still and close. Path length sums every
# frame's noise with a positive sign, so without the deadband a motionless pair
# accumulates phantom speed that grows with the window.
rng = np.random.default_rng(0)
frames = []
for i in range(6):
    jitter_a = tuple(rng.uniform(-2, 2, 2))
    jitter_b = tuple(rng.uniform(-2, 2, 2))
    a = make_person(1, 100, 100, wrist_offset=jitter_a)
    b = make_person(2, 160, 100, wrist_offset=jitter_b)
    frames.append(([a, b], i * 0.5))
results.append(run("keypoint jitter: still pair must not drift into an alert", frames, False))

# A real fight that is not a clean exchange of punches must still fire. This is
# the counterweight to every suppression test above: the motion floor buys its
# false-positive reduction by requiring movement, and the way that goes wrong is
# silently — fights simply stop being reported and nothing in the log says so.
# Tuning MOTION_FLOOR or the weights upward until this test fails means the
# detector has been tuned into uselessness.
SCUFFLE = [0, 45, 6, 47, 4]  # shorter, messier reach than PUNCH_CYCLE
frames = []
for i, reach in enumerate(SCUFFLE):
    a = make_person(1, 100, 100, wrist_offset=(reach, 0))
    b = make_person(2, 170, 100, wrist_offset=(-reach, 0))
    frames.append(([a, b], i * 0.5))
results.append(run("fight: messy scuffle, not clean punches", frames, True))

# The floor has to actually be load-bearing. If MOTION_FLOOR were removed, a
# maximally-close, maximally-contacting, motionless pair must still be unable to
# reach the threshold on the weights alone — belt and braces, since the weights
# and the floor are two independent defences against the same failure.
still_conf = ad.PROXIMITY_WEIGHT * 1.0 + ad.MOTION_WEIGHT * 0.0 + ad.CONTACT_WEIGHT * 1.0
ok = still_conf < ad.FIGHT_THRESHOLD
print(f"{'PASS' if ok else 'FAIL'}  weights alone stop a motionless pair: "
      f"{still_conf:.2f} < {ad.FIGHT_THRESHOLD}")
results.append(ok)

# ---------------------------------------------------------------------------
# Identity persistence: a name learned while the face was visible must survive
# the frames where it isn't — which is most of a real fight.
# ---------------------------------------------------------------------------

ad.reset_state()
people = [make_person(1, 100, 100), make_person(2, 300, 100)]
face_a = ([88, 55, 112, 80], "Shaurya Vatwani")

# Frame 1: face visible, gets matched.
m1 = ad.associate_faces_to_tracks([face_a], people)
# Frame 2: no faces detected at all (head turned away mid-fight).
m2 = ad.associate_faces_to_tracks([], people)
ok = m1[1] == "Shaurya Vatwani" and m2[1] == "Shaurya Vatwani"
print(f"{'PASS' if ok else 'FAIL'}  identity persists when face disappears: "
      f"frame1={m1[1]!r} frame2={m2[1]!r}")
results.append(ok)

# A track that was never recognized must stay Unknown, not inherit a name.
ok = m2[2] == "Unknown"
print(f"{'PASS' if ok else 'FAIL'}  unrecognized track stays Unknown: {m2[2]!r}")
results.append(ok)

# Majority vote: one stray mislabel must not override the consistent one.
ad.reset_state()
for _ in range(3):
    ad.associate_faces_to_tracks([face_a], people)
ad.associate_faces_to_tracks([([88, 55, 112, 80], "Veer")], people)
ok = ad.resolved_identity(1) == "Shaurya Vatwani"
print(f"{'PASS' if ok else 'FAIL'}  majority vote survives one bad frame: "
      f"{ad.resolved_identity(1)!r}")
results.append(ok)

# Identity must be forgotten when the track goes stale — ByteTrack reuses ids,
# so a stale name would eventually be pasted onto a different person.
ad.reset_state()
ad.associate_faces_to_tracks([face_a], people)
ad.update_track_history(people, 0.0)
ad.update_track_history([], 100.0)  # nobody in frame, well past staleness
ok = ad.resolved_identity(1) == "Unknown"
print(f"{'PASS' if ok else 'FAIL'}  identity cleared after track goes stale: "
      f"{ad.resolved_identity(1)!r}")
results.append(ok)

# Fight alert should carry the remembered name even though no face is visible
# in the frames where the fight actually fires.
ad.reset_state()
fired = []
for i, reach in enumerate(PUNCH_CYCLE):
    a = make_person(1, 100, 100, wrist_offset=(reach, 0))
    b = make_person(2, 170, 100, wrist_offset=(-reach, 0))
    t = i * 0.5
    # Face only visible on the very first frame.
    faces = [([88, 55, 112, 80], "Shaurya Vatwani")] if i == 0 else []
    ad.associate_faces_to_tracks(faces, [a, b])
    ad.update_track_history([a, b], t)
    fired += ad.classify_fights([a, b], {}, t)
ok = bool(fired) and fired[0]["persons_involved"][0] == "Shaurya Vatwani"
got = fired[0]["persons_involved"] if fired else None
print(f"{'PASS' if ok else 'FAIL'}  alert names person seen only in frame 1: {got}")
results.append(ok)

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

# Low-confidence torso keypoints must not be scored at all.
ad.reset_state()
fired = []
for i, reach in enumerate(PUNCH_CYCLE):
    a = make_person(1, 100, 100, wrist_offset=(reach, 0), conf=0.1)
    b = make_person(2, 170, 100, wrist_offset=(-reach, 0), conf=0.1)
    t = i * 0.5
    ad.update_track_history([a, b], t)
    fired += ad.classify_fights([a, b], {}, t)
ok = not fired
print(f"{'PASS' if ok else 'FAIL'}  low-confidence pose is not scored: fired={bool(fired)}")
results.append(ok)

# A brand-new track has no motion history and must not fire immediately.
ad.reset_state()
a = make_person(1, 100, 100, wrist_offset=(60, 0))
b = make_person(2, 170, 100, wrist_offset=(-60, 0))
ad.update_track_history([a, b], 0.0)
ad.update_track_history([a, b], 0.5)
immediate = ad.classify_fights([a, b], {}, 0.5)
ok = not immediate
print(f"{'PASS' if ok else 'FAIL'}  immature track cannot fire: fired={bool(immediate)}")
results.append(ok)

# ---------------------------------------------------------------------------
# Real-world calibration.
#
# Every test above this line is synthetic, and all of them passed while the
# detector was shipping broken — twice. They use idealised kinematics, so they
# verify the *logic* (does a fast pair fire, does a still pair not) but say
# nothing about whether the constants match a real human on a real camera.
# SPEED_REFERENCE=1.2 made real punching unreachable and no synthetic test
# noticed, because the synthetic punches were ~3x faster than a person can
# actually move.
#
# These assertions encode measurements taken with tools/measure_motion.py on
# 2026-08-04 (live 640x360 RTSP camera, production 0.5s cadence, one person,
# ~30s per behaviour class) in body-heights/sec of wrist path speed:
#
#     standing/talking/gesturing   median 0.024   p90 0.136   max 0.242
#     throwing real punches        median 0.275   p90 0.366   max 0.443
#
# They fail if a future retune puts the thresholds back outside the range real
# people occupy. Re-measure before changing the numbers here.
# ---------------------------------------------------------------------------

MEASURED_BENIGN_P90 = 0.136
MEASURED_BENIGN_MAX = 0.242
MEASURED_FIGHT_MEDIAN = 0.275
MEASURED_FIGHT_P90 = 0.366


def motion_for(raw_speed, both_moving=False):
    """Normalised motion score for a measured raw wrist speed."""
    weighted = raw_speed if both_moving else 0.7 * raw_speed
    return min(1.0, weighted / ad.SPEED_REFERENCE)


# A real fight must clear the floor. This is the assertion that would have
# caught the 1.0.7 regression: at SPEED_REFERENCE=1.2 a real punch scored 0.16
# against a floor of 0.45 and could never fire.
one_sided = motion_for(MEASURED_FIGHT_MEDIAN)
ok = one_sided >= ad.MOTION_FLOOR
print(f"{'PASS' if ok else 'FAIL'}  real punching clears the floor: "
      f"{one_sided:.3f} >= {ad.MOTION_FLOOR}")
results.append(ok)

# A mutual scuffle — both people moving — must clear it with real margin.
mutual = motion_for(MEASURED_FIGHT_MEDIAN, both_moving=True)
ok = mutual >= ad.MOTION_FLOOR * 1.5
print(f"{'PASS' if ok else 'FAIL'}  mutual fight clears the floor with margin: "
      f"{mutual:.3f} >= {ad.MOTION_FLOOR * 1.5:.3f}")
results.append(ok)

# Ordinary behaviour must stay under it, or the hug false positives come back.
benign = motion_for(MEASURED_BENIGN_P90)
ok = benign < ad.MOTION_FLOOR
print(f"{'PASS' if ok else 'FAIL'}  benign p90 stays under the floor: "
      f"{benign:.3f} < {ad.MOTION_FLOOR}")
results.append(ok)

# The floor has to sit *between* the two measured classes. Outside that gap it
# is either unreachable by real fights or permanently tripped by normal
# movement, which are precisely the two bugs already shipped.
ok = motion_for(MEASURED_BENIGN_P90) < ad.MOTION_FLOOR <= motion_for(MEASURED_FIGHT_MEDIAN)
print(f"{'PASS' if ok else 'FAIL'}  floor sits in the measured gap: "
      f"{motion_for(MEASURED_BENIGN_P90):.3f} < {ad.MOTION_FLOOR} "
      f"<= {motion_for(MEASURED_FIGHT_MEDIAN):.3f}")
results.append(ok)

# SPEED_REFERENCE must be reachable by a real human at all. Anything above the
# measured p90 of real fighting means the top of the scale is unusable.
ok = ad.SPEED_REFERENCE <= MEASURED_FIGHT_P90 * 1.25
print(f"{'PASS' if ok else 'FAIL'}  SPEED_REFERENCE is humanly reachable: "
      f"{ad.SPEED_REFERENCE} <= {MEASURED_FIGHT_P90 * 1.25:.3f} "
      f"(measured fight p90 {MEASURED_FIGHT_P90})")
results.append(ok)

# The deadband must not eat real fighting motion at the production cadence.
# A punch at the measured median covers this much ground per 0.5s sample:
per_segment = MEASURED_FIGHT_MEDIAN * 0.5
ok = per_segment > ad.WRIST_JITTER_DEADBAND * 3
print(f"{'PASS' if ok else 'FAIL'}  deadband does not eat real punches: "
      f"segment {per_segment:.3f} > {ad.WRIST_JITTER_DEADBAND * 3:.3f}")
results.append(ok)

# ---------------------------------------------------------------------------
# End-to-end replay of real recorded behaviour.
#
# The strongest test in this file: real wrist speeds, recorded from the actual
# camera, driven through the real classify_fights() with keypoint confidences
# shaped like the real camera's (shoulders reliable, hips never resolved —
# which is what made MIN_TORSO_KEYPOINTS=2 silently discard every fight).
#
# Unlike the synthetic cases above, this exercises the interaction of all four
# tuned constants at once — SPEED_REFERENCE, MOTION_FLOOR, CONSECUTIVE_HITS and
# MIN_TORSO_KEYPOINTS. Each was individually defensible in 1.0.7 and the
# combination still failed, so testing them together is the point.
# ---------------------------------------------------------------------------

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "measured_wrist_speeds.json"
measured = json.loads(FIXTURE.read_text(encoding="utf-8"))


def camera_shaped_person(track_id, cx, cy, height, wrist_x):
    """A person with this camera's real keypoint availability profile."""
    half_w, half_h = height * 0.25, height / 2
    kp = np.zeros((17, 2), dtype=float)
    kpc = np.zeros(17, dtype=float)
    kp[ad.LEFT_SHOULDER] = [cx - half_w, cy - half_h * 0.3]
    kpc[ad.LEFT_SHOULDER] = 0.86      # measured median
    kp[ad.RIGHT_SHOULDER] = [cx + half_w, cy - half_h * 0.3]
    kpc[ad.RIGHT_SHOULDER] = 0.93     # measured median
    kpc[ad.LEFT_HIP] = 0.03           # measured — effectively never resolved
    kpc[ad.RIGHT_HIP] = 0.06
    kp[ad.LEFT_WRIST] = [wrist_x, cy]
    kpc[ad.LEFT_WRIST] = 0.7
    kp[ad.RIGHT_WRIST] = [wrist_x, cy]
    kpc[ad.RIGHT_WRIST] = 0.7
    return {
        "track_id": track_id,
        "box": [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        "keypoints": kp,
        "kp_conf": kpc,
    }


def replay_measured(raw_speeds, gap_px, label, expect, height=100.0):
    """Drive a recorded speed sequence through the real classifier."""
    ad.reset_state()
    fired = []
    offset_a = offset_b = 0.0
    step = 0.5  # production anomaly cadence
    for i, speed in enumerate(raw_speeds):
        # Alternate direction so this is path length (punch out, retract),
        # which is what limb_speed measures — not one-way drift.
        delta = speed * height * step * (1 if i % 2 == 0 else -1)
        offset_a += delta
        offset_b -= delta
        a = camera_shaped_person(1, 100, 100, height, 100 + offset_a)
        b = camera_shaped_person(2, 100 + gap_px, 100, height, 100 + gap_px + offset_b)
        t = i * step
        ad.update_track_history([a, b], t)
        fired += ad.classify_fights([a, b], {}, t)
    ok = bool(fired) == expect
    conf = f" conf={fired[0]['confidence']}" if fired else ""
    print(f"{'PASS' if ok else 'FAIL'}  replay {label}: fired={bool(fired)}{conf}")
    return ok


benign_speeds = measured["benign_talking_gesturing"]
fight_speeds = measured["real_punching"]

# A single dropped frame mid-fight must not erase real progress. Observed live
# (2026-08-04, real camera, not staged): a pair scored conf=0.641 — clear of
# threshold — and the very next frame for that same pair was gated (a torso
# keypoint dropped) before CONSECUTIVE_HITS was reached. The old behaviour
# (hard .pop on any gated/sub-threshold frame) would let one blurry frame in
# an actual fight erase the streak and prevent the alert; decay lets it survive
# an isolated blip while still fully resetting a genuinely calm pair.
#
# A longer punch cycle (CONSECUTIVE_HITS=3 needs more scoring opportunities
# than the short PUNCH_CYCLE gives once the first 2 frames are spent on track
# maturity), with one frame's confidence collapsed to simulate the dropped
# frame observed live. Progress must be 1, 2, then DECAY to 1 (not reset to
# 0) at the drop, then 2, 3 — firing on the frame right after recovery, one
# frame sooner than a hard reset would allow.
ad.reset_state()
fired = []
LONG_PUNCH_CYCLE = [0, 60, 5, 62, 3, 61, 4, 63, 2, 60]
DROP_AT = 4
for i, reach in enumerate(LONG_PUNCH_CYCLE):
    conf = 0.1 if i == DROP_AT else 1.0  # torso confidence collapses this frame
    a = make_person(1, 100, 100, wrist_offset=(reach, 0), conf=conf)
    b = make_person(2, 170, 100, wrist_offset=(-reach, 0), conf=conf)
    t = i * 0.5
    ad.update_track_history([a, b], t)
    fired += ad.classify_fights([a, b], {}, t)
ok = bool(fired)
print(f"{'PASS' if ok else 'FAIL'}  one dropped frame mid-fight does not erase the streak: fired={bool(fired)}")
results.append(ok)

# But decay must not let a genuinely calm pair drift into firing via noise —
# it should never net-accumulate when it never legitimately scores.
ad.reset_state()
fired = []
still = make_person(1, 100, 100), make_person(2, 170, 100)
weak_still_a = make_person(1, 100, 100, conf=0.1)
weak_still_b = make_person(2, 170, 100, conf=0.1)
for i in range(10):
    a, b = (weak_still_a, weak_still_b) if i % 3 == 0 else still
    t = i * 0.5
    ad.update_track_history([a, b], t)
    fired += ad.classify_fights([a, b], {}, t)
ok = not fired
print(f"{'PASS' if ok else 'FAIL'}  decay does not let a calm pair drift into firing: fired={bool(fired)}")
results.append(ok)

# Real fights must be detected, at a range of separations.
results.append(replay_measured(fight_speeds, 55, "real fight, close", True))
results.append(replay_measured(fight_speeds, 70, "real fight, arm's length", True))
results.append(replay_measured(fight_speeds, 90, "real fight, further apart", True))

# Real benign behaviour must not alert, however close the two people are.
results.append(replay_measured(benign_speeds, 45, "benign gesturing at hugging distance", False))
results.append(replay_measured(benign_speeds, 55, "benign gesturing, close", False))
results.append(replay_measured(benign_speeds, 70, "benign gesturing, conversation", False))
results.append(replay_measured([0.0] * 40, 45, "still pair, embraced", False))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
