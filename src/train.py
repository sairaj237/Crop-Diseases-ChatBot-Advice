import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
import timm
import numpy as np
import mlflow
import mlflow.pytorch
from pathlib import Path

# ------------------
# CONFIG
# ------------------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------
# TRANSFORMS
# ------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------
# DATASETS
# ------------------
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

# ------------------
# BALANCED SAMPLER
# ------------------
targets = np.array(train_ds.targets)
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts

sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------
# MODEL (RESNET18)
# ------------------
model = timm.create_model(
    "resnet18",
    pretrained=True,
    num_classes=NUM_CLASSES
)
model.to(DEVICE)

# ------------------
# LOSS (CLASS WEIGHTED)
# ------------------
loss_weights = torch.tensor(
    class_weights, dtype=torch.float
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ------------------
# TRAINING
# ------------------
mlflow.set_experiment("sugarcane-disease")

with mlflow.start_run(run_name="resnet18_balanced"):
    mlflow.log_params({
        "model": "resnet18",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "balanced_sampler": True,
        "class_weighted_loss": True
    })

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ------------------
        # VALIDATION
        # ------------------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                preds = out.argmax(dim=1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        mlflow.log_metrics({
            "train_loss": train_loss / len(train_loader),
            "val_accuracy": acc,
            "val_precision": precision,
            "val_recall": recall
        }, step=epoch)

        print(
            f"Epoch {epoch+1}: "
            f"acc={acc:.4f}, "
            f"prec={precision:.4f}, "
            f"recall={recall:.4f}"
        )

    # ------------------
    # SAVE FOR DVC + MLFLOW
    # ------------------
    torch.save(model.state_dict(), MODEL_DIR / "model.pth")
    mlflow.pytorch.log_model(model, "model")
