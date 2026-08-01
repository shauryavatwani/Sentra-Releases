"""Registered-people management: reads/writes the real Database/face_embeddings.pkl.

Registration re-uses Formal_Code/face_register.py's own model loading and
embedding logic directly (load_face_model, process_person, normalized_average) —
this module never re-implements face recognition, it only saves uploaded images
into the Faces/<name>/ layout face_register.py already expects, then calls into it.
"""

from __future__ import annotations

import pickle
import re
import sys
import threading
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Formal_Code"))
import face_register  # noqa: E402  (path must be set up first)
import sentra_paths  # noqa: E402

FACES_DIR = sentra_paths.FACES_DIR
DATABASE_FILE = sentra_paths.FACE_EMBEDDINGS_FILE

_model_lock = threading.Lock()
_model = None

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _'-]{0,63}$")


class InvalidNameError(ValueError):
    pass


def validate_name(name: str) -> str:
    name = name.strip()
    if not NAME_PATTERN.match(name):
        raise InvalidNameError(
            "Name must be 1-64 characters: letters, numbers, spaces, hyphens, "
            "apostrophes or underscores only."
        )
    return name


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = face_register.load_face_model()
        return _model


def load_people() -> dict[str, np.ndarray]:
    if not DATABASE_FILE.is_file():
        return {}
    with DATABASE_FILE.open("rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, dict) else {}


def list_names() -> list[str]:
    return sorted(load_people().keys())


def _save_people(data: dict[str, np.ndarray]) -> None:
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATABASE_FILE.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def register_person(name: str, image_bytes_list: list[bytes]) -> dict:
    """Save uploaded images under Faces/<name>/, then embed via face_register.py."""
    name = validate_name(name)
    if not image_bytes_list:
        raise ValueError("At least one image is required.")

    person_dir = FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in person_dir.glob("upload_*.jpg")}
    start = len(existing)
    saved_paths = []
    for i, raw in enumerate(image_bytes_list, start=start):
        path = person_dir / f"upload_{i:03d}.jpg"
        path.write_bytes(raw)
        saved_paths.append(path)

    try:
        model = _get_model()
        embedding, processed, skipped = face_register.process_person(model, person_dir)
    except Exception:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise

    if embedding is None:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise ValueError(
            f"No usable face found in the uploaded image(s) "
            f"({skipped} skipped). Each photo must contain exactly one clear face."
        )

    people = load_people()
    people[name] = embedding
    _save_people(people)

    return {"name": name, "processed": processed, "skipped": skipped}


def delete_person(name: str) -> bool:
    people = load_people()
    if name not in people:
        return False
    del people[name]
    _save_people(people)
    return True
