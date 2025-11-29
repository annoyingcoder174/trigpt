# src/train_face_model.py

from pathlib import Path
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


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


def create_dataloaders(data_root: Path, batch_size: int = 16):
  """
  Expects:
    data_root / "train" / <class_name> / *.jpg
    data_root / "val"   / <class_name> / *.jpg
  """
  data_root = Path(data_root)
  train_dir = data_root / "train"
  val_dir = data_root / "val"

  # Stronger augmentations for better generalization
  train_transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      ),
  ])

  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      ),
  ])

  train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
  val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

  if len(train_dataset) == 0:
    raise RuntimeError(f"No training images found under {train_dir}")
  if len(val_dataset) == 0:
    raise RuntimeError(f"No validation images found under {val_dir}")

  print("Classes (folder names):", train_dataset.classes)
  print("Num train images:", len(train_dataset))
  print("Num val images:", len(val_dataset))

  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
  )
  val_loader = DataLoader(
      val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
  )

  return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int):
  """
  ResNet18 with final FC replaced for our num_classes.
  """
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, num_classes)
  return model


def train_one_epoch(model, loader, device, criterion, optimizer):
  model.train()
  running_loss = 0.0
  running_corrects = 0
  total = 0

  for inputs, labels in loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels).item()
    total += inputs.size(0)

  epoch_loss = running_loss / total
  epoch_acc = running_corrects / total
  return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, device, criterion):
  model.eval()
  running_loss = 0.0
  running_corrects = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels).item()
      total += inputs.size(0)

  epoch_loss = running_loss / total
  epoch_acc = running_corrects / total
  return epoch_loss, epoch_acc


def main():
  data_dir = Path("data/faces")
  models_dir = Path("models")
  models_dir.mkdir(parents=True, exist_ok=True)

  device = get_device()

  train_loader, val_loader, classes = create_dataloaders(
      data_dir,
      batch_size=16,
  )
  num_classes = len(classes)

  model = build_model(num_classes).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

  num_epochs = 25  # tweak if you want more/less training
  best_val_acc = 0.0
  best_state = None

  for epoch in range(num_epochs):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(
        model, train_loader, device, criterion, optimizer
    )
    val_loss, val_acc = eval_one_epoch(
        model, val_loader, device, criterion
    )
    dt = time.time() - t0

    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"- {dt:.1f}s "
        f"- train loss: {train_loss:.4f}, acc: {train_acc:.3f} "
        f"- val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
    )

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_state = model.state_dict()

  if best_state is None:
    best_state = model.state_dict()

  checkpoint_path = models_dir / "face_classifier.pt"
  torch.save(
      {
          "model_state_dict": best_state,
          "classes": classes,  # very important for FaceClassifier
          "best_val_acc": best_val_acc,
      },
      checkpoint_path,
  )

  print("Saved best model to", checkpoint_path)
  print("Best val acc:", best_val_acc)
  print("Classes:", classes)


if __name__ == "__main__":
  main()
