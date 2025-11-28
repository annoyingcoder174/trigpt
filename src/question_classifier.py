# src/question_classifier.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class QuestionTypeClassifier:
    def __init__(self, model_dir: str | Path = "models/question_classifier"):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Question classifier not found at {model_dir}. "
                f"Train it first with train_question_classifier.py."
            )

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Metal) for question classifier 🚀")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for question classifier")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

        # load labels
        labels_path = model_dir / "labels.txt"
        with labels_path.open("r") as f:
            self.labels: List[str] = [line.strip() for line in f if line.strip()]

    def classify(self, question: str) -> Tuple[str, float]:
        """
        Returns (label, confidence)
        """
        inputs = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            conf, idx = torch.max(probs, dim=0)

        label = self.labels[idx.item()]
        return label, conf.item()
