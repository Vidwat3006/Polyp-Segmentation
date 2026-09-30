from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import KvasirDataset


PROJECT_ROOT = Path("..")

KVASIR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "Kvasir-SEG"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "configs"
    / "kvasir_split.json"
)


dataset = KvasirDataset(
    root_dir=KVASIR_ROOT,
    split_file=SPLIT_FILE,
    split="train"
)


def denormalize(image):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)


# ------------------------------------------------------------
# Get the same image multiple times
# ------------------------------------------------------------

samples = []

for _ in range(4):
    image, mask = dataset[0]

    image = denormalize(image)

    image = (
        image
        .permute(1, 2, 0)
        .numpy()
    )

    mask = mask.squeeze(0).numpy()

    samples.append((image, mask))


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

fig, axes = plt.subplots(
    4,
    3,
    figsize=(12, 16)
)

for row, (image, mask) in enumerate(samples):

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(
        f"Augmented Image {row + 1}"
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(
        mask,
        cmap="gray"
    )
    axes[row, 1].set_title("Mask")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(image)
    axes[row, 2].imshow(
        mask,
        alpha=0.4,
        cmap="gray"
    )
    axes[row, 2].set_title("Overlay")
    axes[row, 2].axis("off")

plt.tight_layout()
plt.show()
