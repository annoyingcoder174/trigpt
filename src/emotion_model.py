# src/emotion_model.py

from __future__ import annotations

from pathlib import Path
import io
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Metal) for emotion model 🚀")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU for emotion model")
        return torch.device("cpu")


class EmotionClassifier:
    def __init__(self, checkpoint_path: Path | str = "models/emotion_classifier.pt", device: torch.device | None = None):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Emotion classifier checkpoint not found at {checkpoint_path}. Train it first."
            )

        self.device = device or get_device()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        num_classes = len(self.classes)

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

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
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def predict_pil(self, img: Image.Image) -> Tuple[str, float]:
        x = self._preprocess_pil(img)

        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        label = self.classes[idx.item()]
        return label, conf.item()

    def predict_from_bytes(self, image_bytes: bytes) -> Tuple[str, float]:
        img = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil(img)

    def predict_from_path(self, image_path: Path | str) -> Tuple[str, float]:
        img = Image.open(image_path)
        return self.predict_pil(img)
