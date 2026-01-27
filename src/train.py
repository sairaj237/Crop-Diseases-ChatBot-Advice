import torch
import timm
import mlflow
import mlflow.pytorch
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/model")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = torch.load(DATA_DIR / "train.pt")
val_ds = torch.load(DATA_DIR / "val.pt")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

num_classes = len(train_ds.dataset.classes)

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

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
    correct = 0
    total = 0

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

print("Training complete.")
