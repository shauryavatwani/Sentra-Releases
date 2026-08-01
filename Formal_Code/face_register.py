"""Build the face-embedding database used by face_recognition.py.

Expected layout:
Shark_Tank/
├── Faces/
│   ├── Person Name/
│   │   ├── front.jpg
│   │   ├── left.jpg
│   │   └── right.jpg
├── Database/
└── Formal_Code/
    └── face_register.py

Each subfolder in Faces is treated as one person's label. Images must contain
exactly one clear face. The script stores one averaged, normalized 512-value
embedding per person in Database/face_embeddings.pkl.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import sentra_paths


# Paths come from sentra_paths so registration writes to the same place the
# engine reads from, including on an installed Windows build where that is
# ProgramData rather than the (read-only) program folder.
FACES_DIR = sentra_paths.FACES_DIR
DATABASE_DIR = sentra_paths.DATABASE_DIR
DATABASE_FILE = sentra_paths.FACE_EMBEDDINGS_FILE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECTION_SIZE = (640, 640)


def load_face_model() -> FaceAnalysis:
    """Load InsightFace with detection and recognition enabled."""
    print("Loading InsightFace model...")
    model = FaceAnalysis(
        root=str(sentra_paths.insightface_root()),
        providers=["CPUExecutionProvider"],
    )
    model.prepare(ctx_id=0, det_size=DETECTION_SIZE)
    print("Model loaded.\n")
    return model


def image_paths_for(person_dir: Path) -> list[Path]:
    """Return supported image files in a person's folder, in stable order."""
    return sorted(
        path
        for path in person_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_average(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average embeddings and normalize the result for cosine similarity."""
    average = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm = np.linalg.norm(average)
    if norm == 0:
        raise ValueError("The averaged embedding had zero length.")
    return average / norm


def process_person(model: FaceAnalysis, person_dir: Path) -> tuple[np.ndarray | None, int, int]:
    """Create one representative embedding from all valid images for a person."""
    embeddings: list[np.ndarray] = []
    processed = 0
    skipped = 0

    print(f"Processing {person_dir.name}...")
    image_paths = image_paths_for(person_dir)

    if not image_paths:
        print("  No supported image files found.")
        return None, processed, skipped

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Skipped {image_path.name}: unreadable image.")
            skipped += 1
            continue

        faces = model.get(image)
        if len(faces) != 1:
            reason = "no face found" if not faces else f"{len(faces)} faces found"
            print(f"  Skipped {image_path.name}: {reason}; exactly one is required.")
            skipped += 1
            continue

        # normed_embedding is already unit-normalized. Copy it to avoid
        # retaining model-owned data in the final database.
        embeddings.append(np.asarray(faces[0].normed_embedding, dtype=np.float32).copy())
        processed += 1
        print(f"  Added {image_path.name}")

    if not embeddings:
        print(f"  No valid images registered for {person_dir.name}.\n")
        return None, processed, skipped

    print(f"  Registered {person_dir.name} from {len(embeddings)} image(s).\n")
    return normalized_average(embeddings), processed, skipped


def main() -> int:
    if not FACES_DIR.is_dir():
        print(f"Error: Faces folder not found: {FACES_DIR}")
        return 1

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    model = load_face_model()

    database: dict[str, np.ndarray] = {}
    total_processed = 0
    total_skipped = 0

    person_dirs = sorted(path for path in FACES_DIR.iterdir() if path.is_dir())
    if not person_dirs:
        print(f"Error: no person folders were found in {FACES_DIR}")
        return 1

    print(f"Found {len(person_dirs)} person folder(s).\n")
    for person_dir in person_dirs:
        embedding, processed, skipped = process_person(model, person_dir)
        total_processed += processed
        total_skipped += skipped
        if embedding is not None:
            database[person_dir.name] = embedding

    if not database:
        print("Error: no people were registered; the database was not written.")
        return 1

    with DATABASE_FILE.open("wb") as database_handle:
        pickle.dump(database, database_handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("=" * 60)
    print("FACE DATABASE CREATED")
    print("=" * 60)
    print(f"People registered : {len(database)}")
    print(f"Images processed  : {total_processed}")
    print(f"Images skipped    : {total_skipped}")
    print(f"Saved to          : {DATABASE_FILE}")
    print("\nReady for face_recognition.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
