# src/predict_face.py

from pathlib import Path
import argparse

from .uniface_face_classifier import FaceClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Predict identity (PTri / Lanh / MTuan / BHa / PTri's Muse / strangers / ...) from an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    # checkpoint_path is kept for API compatibility but ignored by UniFace backend
    classifier = FaceClassifier(checkpoint_path="models/face_classifier.pt")

    class_name, confidence = classifier.predict_from_path(image_path)
    print(f"\nPrediction: {class_name}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Done.")


if __name__ == "__main__":
    main()
