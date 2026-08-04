"""Synthetic tests for anomaly_detection.py — no camera or model needed."""
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

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
