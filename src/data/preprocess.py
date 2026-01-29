from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

SPLIT = 0.2
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    classes = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    assert classes, "No class folders found in data/raw"

    for cls in classes:
        images = [p for p in cls.iterdir() if p.suffix.lower() in IMG_EXTS]
        assert images, f"No images in {cls.name}"

        train_imgs, val_imgs = train_test_split(
            images, test_size=SPLIT, random_state=SEED
        )

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_cls = OUT_DIR / split / cls.name
            out_cls.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                shutil.copy(img, out_cls / img.name)

    print("Preprocessing done. Train/Val folders created.")

if __name__ == "__main__":
    main()
