from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io
import os

import cv2
import numpy as np
from PIL import Image

# --- Optional HEIC support (same style as vision_model.py) ---
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    print("[UniFaceFaceClassifier] HEIC/HEIF support enabled via pillow-heif.")
except Exception:
    print("[UniFaceFaceClassifier] pillow-heif not installed; HEIC/HEIF images may not work.")

# UniFace imports
from uniface import RetinaFace, ArcFace, compute_similarity


@dataclass
class FaceDBEntry:
    label: str
    mean_embedding: np.ndarray           # (D,)
    num_samples: int
    raw_embeddings: np.ndarray           # (N, D)


class FaceClassifier:
    """
    Drop-in replacement for src/vision_model.FaceClassifier,
    implemented using UniFace (RetinaFace + ArcFace).

    Public interface kept the same on purpose:

        classifier = FaceClassifier(checkpoint_path="models/face_classifier.pt")
        label, conf = classifier.predict_from_bytes(image_bytes)
        label, conf = classifier.predict_from_path(path)
        classifier.classes  # -> list of labels

    Internally:

    - Builds a face database from:
        data/faces/train/<person_name>/*.(jpg|jpeg|png|heic...)
        data/faces/val/<person_name>/*.(jpg|jpeg|png|heic...)
    - Each person gets a **mean embedding** so people with 90+ images
      don't overwhelm those with 20 images.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = "models/face_classifier.pt",
        faces_dirs: Optional[List[Path | str]] = None,
        min_detection_conf: float = 0.6,
        similarity_threshold: float = 0.45,
        max_images_per_person: int = 200,
    ) -> None:
        """
        Args:
            checkpoint_path: Ignored, kept only for API compatibility with old code.
            faces_dirs: Root folders of identity images.
                        Default: ["data/faces/train", "data/faces/val"] if they exist.
            min_detection_conf: Minimum RetinaFace confidence to accept a face.
            similarity_threshold: Minimum cosine similarity (0–1) to accept a match.
                                  Below this → 'strangers'.
            max_images_per_person: Hard cap on images per person to keep startup sane.
        """
        if faces_dirs is None:
            faces_dirs = [
                Path("data/faces/train"),
                Path("data/faces/val"),
            ]

        self.min_detection_conf = float(min_detection_conf)

        # Allow overriding threshold by env var if you ever want to tweak quickly:
        env_thresh = os.getenv("UNIFACE_SIM_THRESH")
        if env_thresh is not None:
            try:
                similarity_threshold = float(env_thresh)
            except Exception:
                pass

        # Default is a bit softer than before to be less “strangers”-happy
        self.similarity_threshold = float(similarity_threshold)  # e.g. 0.35–0.45
        self.max_images_per_person = int(max_images_per_person)

        print(
            f"[UniFaceFaceClassifier] Init with similarity_threshold="
            f"{self.similarity_threshold:.2f}, min_det_conf={self.min_detection_conf:.2f}"
        )

        # UniFace models
        self.detector = RetinaFace()
        self.recognizer = ArcFace()

        # Face database
        self._entries: Dict[str, FaceDBEntry] = {}
        self._labels: List[str] = []

        # Build embeddings DB at startup
        self._build_database([Path(p) for p in faces_dirs])

    # ------------------------------------------------------------------
    # Public attributes expected by existing code
    # ------------------------------------------------------------------
    @property
    def classes(self) -> List[str]:
        """List of known identity labels (folder names)."""
        return list(self._labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _iter_person_images(self, root: Path) -> Dict[str, List[Path]]:
        """
        Scan a root directory like data/faces/train for subfolders:

            root/<label>/*.{jpg,jpeg,png,bmp,webp,heic,HEIC}
        """
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".HEIC"}
        people: Dict[str, List[Path]] = {}

        if not root.exists():
            return people

        for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            label = person_dir.name
            img_paths: List[Path] = []
            for path in sorted(person_dir.rglob("*")):
                if path.is_file() and path.suffix in exts:
                    img_paths.append(path)
            if img_paths:
                people.setdefault(label, []).extend(img_paths)

        return people

    @staticmethod
    def _path_to_bgr(path: Path) -> Optional[np.ndarray]:
        """
        Load an image from disk (supports HEIC via Pillow) and return BGR np.ndarray.
        """
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                arr = np.asarray(img)
        except Exception:
            return None

        # Pillow gives RGB; UniFace expects BGR (cv2 style)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bytes_to_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode raw bytes into a BGR image (using Pillow + optional pillow-heif).
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                arr = np.asarray(img)
        except Exception:
            return None

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _build_database(self, roots: List[Path]) -> None:
        """
        Build per-person embeddings from images on disk.

        This runs once at startup. If you add new people or new images,
        **restart the backend** so the database rebuilds.

        Logic:
        - For each folder <label>, collect images from all roots.
        - For each image, run RetinaFace (max_num=1) and ArcFace to get embedding.
        - L2-normalize each embedding (ArcFace already does this, we enforce again).
        - Average per person → mean embedding.
        """
        label_to_embs: Dict[str, List[np.ndarray]] = {}

        print("[UniFaceFaceClassifier] Building face DB...")
        for root in roots:
            print(f"  scanning root: {root}")
            people = self._iter_person_images(root)
            for label, paths in people.items():
                embs = label_to_embs.setdefault(label, [])
                for path in paths:
                    if len(embs) >= self.max_images_per_person:
                        break

                    img_bgr = self._path_to_bgr(path)
                    if img_bgr is None:
                        continue

                    # Only need a single face per training image
                    faces = self.detector.detect(img_bgr, max_num=1)
                    if not faces:
                        continue

                    face = faces[0]
                    conf = float(face.get("confidence", 0.0))
                    if conf < self.min_detection_conf:
                        continue

                    landmarks = face.get("landmarks", None)
                    if landmarks is None or np.asarray(landmarks).size == 0:
                        continue

                    try:
                        emb = self.recognizer.get_normalized_embedding(
                            img_bgr,
                            np.asarray(landmarks, dtype=np.float32),
                        )
                    except Exception:
                        continue

                    emb = np.asarray(emb).ravel().astype("float32")
                    norm = np.linalg.norm(emb)
                    if norm == 0:
                        continue
                    emb = emb / norm
                    embs.append(emb)

                # Aggregate into mean embeddings
        for label, embs in label_to_embs.items():
            if not embs:
                continue

            # ⚠️ Do NOT treat "strangers" as a real identity.
            # It should only be a fallback label when similarity is too low.
            if label.lower() == "strangers":
                print("[UniFaceFaceClassifier] Skipping 'strangers' as a trained identity.")
                continue

            mat = np.stack(embs).astype("float32")  # (N, D)


            mean_emb = mat.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm == 0:
                continue
            mean_emb = mean_emb / norm

            entry = FaceDBEntry(
                label=label,
                mean_embedding=mean_emb,
                num_samples=len(embs),
                raw_embeddings=mat,
            )
            self._entries[label] = entry

        self._labels = sorted(self._entries.keys())

        print(
            f"[UniFaceFaceClassifier] Built face DB with "
            f"{len(self._labels)} identities."
        )
        for lbl in self._labels:
            n = self._entries[lbl].num_samples
            print(f"  - {lbl}: {n} images")

    # ------------------------------------------------------------------
    # Core prediction logic
    # ------------------------------------------------------------------
    def _predict_from_bgr(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Detect the *most salient* face in an image and return (label, confidence).

        - If no DB → always 'strangers'.
        - If no face detected → 'strangers'.
        - If similarity < similarity_threshold → 'strangers'.
        """
        if not self._entries:
            return "strangers", 0.0

        faces = self.detector.detect(image_bgr, max_num=1)
        if not faces:
            return "strangers", 0.0

        face = faces[0]
        det_conf = float(face.get("confidence", 0.0))
        if det_conf < self.min_detection_conf:
            return "strangers", det_conf

        landmarks = face.get("landmarks", None)
        if landmarks is None or np.asarray(landmarks).size == 0:
            return "strangers", det_conf

        try:
            emb = self.recognizer.get_normalized_embedding(
                image_bgr,
                np.asarray(landmarks, dtype=np.float32),
            )
        except Exception:
            return "strangers", 0.0

        emb = np.asarray(emb).ravel().astype("float32")
        norm = np.linalg.norm(emb)
        if norm == 0:
            return "strangers", 0.0
        emb = emb / norm  # ensure normalized

        # Compare with each person's mean embedding
        best_label: Optional[str] = None
        best_sim: float = -1.0

        for label, entry in self._entries.items():
            sim = float(compute_similarity(emb, entry.mean_embedding, normalized=True))
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_label is None:
            return "strangers", 0.0

        # Cosine similarity in [-1, 1]; clamp to [0, 1] for a confidence-like number
        sim_clamped = max(0.0, min(1.0, best_sim))

        # Debug print for your own testing – you can comment out later.
        print(
            f"[UniFaceFaceClassifier] Prediction – best_label={best_label}, "
            f"sim={sim_clamped:.3f}, threshold={self.similarity_threshold:.3f}"
        )

        if sim_clamped < self.similarity_threshold:
            # Too low → treat as strangers
            return "strangers", sim_clamped

        return best_label, sim_clamped

    # ------------------------------------------------------------------
    # Public methods expected by existing code
    # ------------------------------------------------------------------
    def predict_from_bytes(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Main method used by the API:
          - /predict_face
          - /secure_ask
          - /live_detect (after cropping)
          - Any future vision endpoints.

        It takes raw image bytes, detects the main face, and returns:
           (identity_label, confidence_0_to_1)
        """
        image_bgr = self._bytes_to_bgr(image_bytes)
        if image_bgr is None:
            return "strangers", 0.0
        return self._predict_from_bgr(image_bgr)

    def predict_from_path(self, image_path: Path | str) -> Tuple[str, float]:
        """
        CLI-friendly method, used by src/predict_face.py.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image_bgr = self._path_to_bgr(path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")

        return self._predict_from_bgr(image_bgr)
