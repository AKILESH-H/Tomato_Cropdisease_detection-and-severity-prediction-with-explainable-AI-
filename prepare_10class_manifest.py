
from pathlib import Path
import csv
import random
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = ROOT / "PlantVillage-Dataset" / "raw" / "color"
OUTPUT = ROOT / "results" / "disease_10class_manifest.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {IMAGE_ROOT}")

    rng = random.Random(SEED)
    rows = []
    counts = Counter()

    classes = sorted(
        folder for folder in IMAGE_ROOT.iterdir()
        if folder.is_dir() and folder.name.startswith("Tomato___")
    )

    if len(classes) != 10:
        raise RuntimeError(
            f"Expected 10 tomato classes, but found {len(classes)}. "
            f"Found: {[folder.name for folder in classes]}"
        )

    for class_folder in classes:
        images = sorted(
            path for path in class_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        rng.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        split_groups = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, split_images in split_groups.items():
            for image_path in split_images:
                rows.append({
                    "image_path": str(image_path.resolve()),
                    "class_name": class_folder.name,
                    "split": split,
                })
                counts[(split, class_folder.name)] += 1

        print(
            f"{class_folder.name}: total={total}, "
            f"train={len(split_groups['train'])}, "
            f"validation={len(split_groups['validation'])}, "
            f"test={len(split_groups['test'])}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "class_name", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nManifest created successfully.")
    print(f"Classes: {len(classes)}")
    print(f"Total images: {len(rows)}")
    print(f"Training images: {sum(v for (s, _), v in counts.items() if s == 'train')}")
    print(f"Validation images: {sum(v for (s, _), v in counts.items() if s == 'validation')}")
    print(f"Test images: {sum(v for (s, _), v in counts.items() if s == 'test')}")
    print(f"Saved to: {OUTPUT}")
    print("Original images were not moved, copied, or deleted.")

if __name__ == "__main__":
    main()
