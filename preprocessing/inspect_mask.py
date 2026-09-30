from pathlib import Path
from PIL import Image
import numpy as np


DATASET_DIR = Path("./datasets")


def inspect_mask(mask_path):
    mask = Image.open(mask_path).convert("RGB")
    arr = np.array(mask)

    print(f"\nFile: {mask_path.name}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

    # Find unique RGB colors
    pixels = arr.reshape(-1, 3)
    unique_colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True
    )

    print(f"Unique RGB colors: {len(unique_colors)}")

    # Show the most common colors
    indices = np.argsort(counts)[::-1]

    print("\nMost common colors:")

    for i in indices[:20]:
        color = tuple(unique_colors[i])
        count = counts[i]
        percentage = (count / len(pixels)) * 100

        print(
            f"  {color} -> "
            f"{count} pixels "
            f"({percentage:.2f}%)"
        )


# --------------------------------------------------
# Kvasir
# --------------------------------------------------

print("=" * 60)
print("KVASIR-SEG MASK")
print("=" * 60)

kvasir_mask_dir = DATASET_DIR / "Kvasir-SEG" / "masks"

kvasir_masks = sorted(
    kvasir_mask_dir.glob("*")
)

inspect_mask(kvasir_masks[0])


# --------------------------------------------------
# CVC
# --------------------------------------------------

print("\n" + "=" * 60)
print("CVC-CLINICDB MASK")
print("=" * 60)

cvc_mask_dir = DATASET_DIR / "CVC-ClinicDB" / "Ground Truth"

cvc_masks = sorted(
    cvc_mask_dir.glob("*")
)

inspect_mask(cvc_masks[0])


