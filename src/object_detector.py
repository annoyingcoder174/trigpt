# src/object_detector.py

from io import BytesIO
from typing import List, Dict

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


COCO_LABELS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]


class ObjectDetector:
    def __init__(self, device: str | None = None, score_thresh: float = 0.5):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = torch.device(device)
        self.score_thresh = score_thresh

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    @torch.inference_mode()
    def detect(
        self,
        image_bytes: bytes,
        max_dets: int = 10,
    ) -> List[Dict]:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = img.size

        tensor = self.transform(img).to(self.device)
        outputs = self.model([tensor])[0]

        boxes = outputs["boxes"].cpu().tolist()
        scores = outputs["scores"].cpu().tolist()
        labels = outputs["labels"].cpu().tolist()

        detections: List[Dict] = []

        for box, score, label_id in zip(boxes, scores, labels):
            if score < self.score_thresh:
                continue

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            # normalise 0–1 so frontend can scale to any size
            detections.append({
                "label": COCO_LABELS[label_id],
                "score": float(score),
                "x": float(x1 / width),
                "y": float(y1 / height),
                "w": float(w / width),
                "h": float(h / height),
            })

            if len(detections) >= max_dets:
                break

        return detections
