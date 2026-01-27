import torch
import timm
import mlflow
import mlflow.pytorch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models/model")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms (same as notebook)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)

val_size = int(len(full_ds) * VAL_SPLIT)
train_size = len(full_ds) - val_size

train_ds, val_ds = random_split(
    full_ds,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

num_classes = len(full_ds.classes)

# Model
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# MLflow
mlflow.start_run(run_name="efficientnet_b0")

mlflow.log_param("epochs", EPOCHS)
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("lr", LR)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "model.pth")

mlflow.pytorch.log_model(model, "model")
mlflow.end_run()

print("Training complete. Model saved.")
