# src/train_emotion_model.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Metal) 🚀")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = 16,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], list[str]]:
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    }

    image_datasets = {
        "train": datasets.ImageFolder(root=str(train_dir), transform=data_transforms["train"]),
        "val": datasets.ImageFolder(root=str(val_dir), transform=data_transforms["val"]),
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"))
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    print("Emotion classes:", class_names)

    return dataloaders, dataset_sizes, class_names


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    device: torch.device,
    num_epochs: int = 10,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if dataset_sizes[phase] > 0:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                epoch_loss = 0.0
                epoch_acc = 0.0

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f"\nBest val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    train_dir = Path("data/emotions/train")
    val_dir = Path("data/emotions/val")

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("Emotion dataset folders not found. Create data/emotions/train and data/emotions/val.")

    device = get_device()

    dataloaders, dataset_sizes, class_names = create_dataloaders(train_dir, val_dir, batch_size=16)

    if dataset_sizes["train"] == 0 or dataset_sizes["val"] == 0:
        raise SystemExit("Train/val folders are empty. Add images for each emotion before training.")

    # Build model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    model = train_model(model, dataloaders, dataset_sizes, device, num_epochs=10)

    # Save checkpoint
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    checkpoint_path = save_dir / "emotion_classifier.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": class_names,
    }
    torch.save(checkpoint, checkpoint_path)

    print(f"\n✅ Training complete. Emotion model saved to {checkpoint_path}")
    print("Classes order:", class_names)


if __name__ == "__main__":
    main()
