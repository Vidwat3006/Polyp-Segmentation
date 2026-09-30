from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import (
    KvasirDataset,
    CVCClinicDBDataset
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("..")

KVASIR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "Kvasir-SEG"
)

CVC_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "CVC-ClinicDB"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "configs"
    / "kvasir_split.json"
)


# ============================================================
# Kvasir Training Dataset
# ============================================================

train_dataset = KvasirDataset(
    root_dir=KVASIR_ROOT,
    split_file=SPLIT_FILE,
    split="train"
)


# ============================================================
# Kvasir Validation Dataset
# ============================================================

val_dataset = KvasirDataset(
    root_dir=KVASIR_ROOT,
    split_file=SPLIT_FILE,
    split="validation"
)


# ============================================================
# CVC Dataset
# ============================================================

cvc_dataset = CVCClinicDBDataset(
    root_dir=CVC_ROOT
)


# ============================================================
# Dataset sizes
# ============================================================

print("=" * 60)
print("DATASET SIZES")
print("=" * 60)

print("Kvasir training   :", len(train_dataset))
print("Kvasir validation :", len(val_dataset))
print("CVC test          :", len(cvc_dataset))


# ============================================================
# Get samples
# ============================================================

image, mask = train_dataset[0]

print("\n" + "=" * 60)
print("KVASIR SAMPLE")
print("=" * 60)

print("Image shape :", image.shape)
print("Image dtype :", image.dtype)

print("Mask shape  :", mask.shape)
print("Mask dtype  :", mask.dtype)

print(
    "Mask values :",
    torch.unique(mask)
)


# ============================================================
# Validation sample
# ============================================================

image_val, mask_val = val_dataset[0]

print("\n" + "=" * 60)
print("VALIDATION SAMPLE")
print("=" * 60)

print("Image shape :", image_val.shape)
print("Mask shape  :", mask_val.shape)
print("Mask values :", torch.unique(mask_val))


# ============================================================
# CVC sample
# ============================================================

image_cvc, mask_cvc = cvc_dataset[0]

print("\n" + "=" * 60)
print("CVC SAMPLE")
print("=" * 60)

print("Image shape :", image_cvc.shape)
print("Mask shape  :", mask_cvc.shape)
print("Mask values :", torch.unique(mask_cvc))


# ============================================================
# Visualization helper
# ============================================================

def denormalize(image):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image * std + mean

    image = image.clamp(0, 1)

    return image


# ============================================================
# Visualize
# ============================================================

image_display = denormalize(image)

image_display = (
    image_display
    .permute(1, 2, 0)
    .numpy()
)

mask_display = (
    mask.squeeze(0)
    .numpy()
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_display)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_display, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_display)
plt.imshow(
    mask_display,
    alpha=0.4,
    cmap="gray"
)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()