from pathlib import Path
import random
import json


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("../datasets")
CONFIG_DIR = Path(".")

SEED = 42
TRAIN_RATIO = 0.80


# ============================================================
# Find Kvasir images
# ============================================================

image_dir = DATASET_DIR / "Kvasir-SEG" / "images"

image_files = sorted([
    p.name
    for p in image_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])


print("Total Kvasir images:", len(image_files))


# ============================================================
# Reproducible shuffle
# ============================================================

random.seed(SEED)

random.shuffle(image_files)


# ============================================================
# Train / validation split
# ============================================================

train_size = int(
    len(image_files) * TRAIN_RATIO
)

train_files = image_files[:train_size]
val_files = image_files[train_size:]


# ============================================================
# Verify
# ============================================================

print("Training images  :", len(train_files))
print("Validation images:", len(val_files))

print("\nSeed:", SEED)
print("Train ratio:", TRAIN_RATIO)


# ============================================================
# Save split
# ============================================================

split = {
    "seed": SEED,
    "train_ratio": TRAIN_RATIO,
    "dataset": "Kvasir-SEG",
    "train": train_files,
    "validation": val_files
}


output_path = CONFIG_DIR / "kvasir_split.json"

with open(output_path, "w") as f:
    json.dump(split, f, indent=4)


print("\nSaved split to:")
print(output_path.resolve())