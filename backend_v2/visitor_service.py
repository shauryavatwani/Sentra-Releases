"""Turning a photo taken at the gate into a visit request.

The counterpart of people_store.py: that module enrols permanent people from
curated photos into ``face_embeddings.pkl``; this one embeds a single one-shot
gate capture and files it as a *pending* visit in ``visitor_store``.

Both use the exact same InsightFace pipeline — this module calls
``face_register.load_face_model`` and reads ``normed_embedding`` off the
detected face, the same values ``process_person`` averages. Nothing about the
recognition maths is re-implemented or loosened for visitors; a visitor is
matched against the same 0.45 cosine threshold as anyone else.

The one real difference is how strict the capture has to be. Enrolment can ask
for three good photos and reject anything imperfect, because the person is
sitting down to be enrolled. A visitor is standing at a gate with a queue
behind them, so the failure messages here have to say what to fix ("no face
found — move closer") rather than just refusing.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Formal_Code"))
import face_register  # noqa: E402  (path must be set up first)
import sentra_paths  # noqa: E402
import visitor_store  # noqa: E402

VISITORS_DIR = sentra_paths.VISITORS_DIR

_model_lock = threading.Lock()
_model = None


class CaptureError(ValueError):
    """The photo could not be used — the message is shown to the guard."""


def _get_model():
    """The InsightFace model, loaded once and shared.

    Deliberately a separate handle from people_store's: both call
    ``face_register.load_face_model()``, and loading it twice costs memory but
    keeps registration and gate capture from blocking each other behind one
    lock while a guard is waiting at the gate.
    """
    global _model
    with _model_lock:
        if _model is None:
            _model = face_register.load_face_model()
        return _model


def embed_capture(image_bytes: bytes) -> np.ndarray:
    """Extract one face embedding from a gate photo.

    Requires exactly one face, matching face_register.py's rule. A frame with
    two people in it is ambiguous — embedding the wrong one would enrol the
    person standing behind the visitor, and nothing downstream could tell.
    """
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise CaptureError("That photo could not be read. Try taking it again.")

    faces = _get_model().get(image)
    if not faces:
        raise CaptureError(
            "No face found in the photo. Ask the visitor to look at the camera "
            "and move closer, then retake it."
        )
    if len(faces) > 1:
        raise CaptureError(
            f"{len(faces)} faces found in the photo. Only the visitor should be "
            "in frame — retake it with everyone else out of shot."
        )

    return np.asarray(faces[0].normed_embedding, dtype=np.float32).copy()


def _save_photo(visitor_id: str, image_bytes: bytes) -> str:
    VISITORS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{visitor_id}.jpg"
    (VISITORS_DIR / filename).write_bytes(image_bytes)
    return filename


def photo_path(photo_file: str) -> Path | None:
    """Resolve a stored photo filename, refusing anything outside Visitors/.

    The filename comes from the database rather than the request, but this is
    the function that turns a string into a file read, so it is the right place
    to make path traversal impossible regardless of how the value got there.
    """
    if not photo_file:
        return None
    candidate = (VISITORS_DIR / photo_file).resolve()
    if candidate.parent != VISITORS_DIR.resolve() or not candidate.is_file():
        return None
    return candidate


def create_visit_request(
    *,
    name: str,
    image_bytes: bytes,
    duration_minutes: int,
    requested_by: str,
    purpose: str = "",
    host: str = "",
) -> dict:
    """Embed the gate photo and file a pending visit request.

    The photo is written only after the embedding succeeds, so a rejected
    capture never leaves an orphan file behind — the same cleanup discipline
    people_store.register_person uses.
    """
    name = (name or "").strip()
    if not name:
        raise CaptureError("Enter the visitor's name.")
    if not image_bytes:
        raise CaptureError("Take a photo of the visitor first.")

    duration_minutes = visitor_store.validate_duration(duration_minutes)
    embedding = embed_capture(image_bytes)

    visitor = visitor_store.create_visitor(
        name=name,
        embedding=embedding,
        photo_file="",  # filled in below, once we know the generated id
        duration_minutes=duration_minutes,
        requested_by=requested_by,
        purpose=purpose,
        host=host,
    )

    try:
        filename = _save_photo(visitor["id"], image_bytes)
    except OSError as exc:
        visitor_store.delete_visitor(visitor["id"])
        raise CaptureError(f"Could not save the visitor's photo: {exc}") from exc

    visitor_store.set_photo_file(visitor["id"], filename)
    return visitor_store.get_visitor(visitor["id"])
