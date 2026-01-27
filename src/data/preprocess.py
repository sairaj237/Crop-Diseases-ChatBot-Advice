import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from pathlib import Path

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

IMG_SIZE = 224
VAL_SPLIT = 0.2
SEED = 42

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def main():
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(train_ds, OUT_DIR / "train.pt")
    torch.save(val_ds, OUT_DIR / "val.pt")

    print("Preprocessing done.")

if __name__ == "__main__":
    main()
