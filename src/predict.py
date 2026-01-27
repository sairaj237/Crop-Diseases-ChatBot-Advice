import torch
import timm
from torchvision import transforms
from PIL import Image
from pathlib import Path

MODEL_PATH = Path("models/model/model.pth")
CLASS_NAMES = ["Healthy", "Red Rot", "Leaf Blight", "Rust", "Mosaic"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(CLASS_NAMES)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()

    return CLASS_NAMES[pred]
