from pathlib import Path

DATA_DIR = Path("data/raw")

if __name__ == "__main__":
    assert DATA_DIR.exists(), "data/raw does not exist"

    classes = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    assert len(classes) > 1, "Need at least 2 class folders"

    for cls in classes:
        imgs = list((DATA_DIR / cls).glob("*"))
        assert len(imgs) > 0, f"No images in class {cls}"

    print(f"Dataset OK. Classes found: {classes}")
