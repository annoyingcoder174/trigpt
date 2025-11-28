# src/vision_model.py

from pathlib import Path
import io

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Metal) 🚀")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


class FaceClassifier:
    def __init__(self, checkpoint_path: Path | str = "models/face_classifier.pt", device: torch.device | None = None):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. Train the model first."
            )

        self.device = device or get_device()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        num_classes = len(self.classes)

        # same architecture as training
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model

        # preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _preprocess_pil(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        tensor = self.transform(img).unsqueeze(0)  # batch dim
        return tensor.to(self.device)

    def predict_pil(self, img: Image.Image) -> tuple[str, float]:
        """
        Predict class name and confidence for a PIL image.
        Returns (class_name, confidence_0_to_1).
        """
        x = self._preprocess_pil(img)

        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        class_name = self.classes[pred_idx.item()]
        confidence = conf.item()
        return class_name, confidence

    def predict_from_bytes(self, image_bytes: bytes) -> tuple[str, float]:
        """
        For use in API: take raw bytes, open as image, run prediction.
        """
        img = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil(img)

    def predict_from_path(self, image_path: Path | str) -> tuple[str, float]:
        img = Image.open(image_path)
        return self.predict_pil(img)
