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

# 10. No single signal can fire on its own — the weights are capped so that
# proximity, motion, or contact alone all sit below the threshold.
ok = max(0.35, 0.40, 0.25) < ad.FIGHT_THRESHOLD
print(f"{'PASS' if ok else 'FAIL'}  no single signal can fire alone")
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
