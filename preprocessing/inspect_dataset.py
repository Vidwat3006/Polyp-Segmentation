from pathlib import Path
from PIL import Image
import cv2



# import os

# path = r"../datasets/Kvasir-SEG/images"

# print("Current directory:", os.getcwd())
# print("Path exists:", os.path.exists(path))
# print("Absolute path:", os.path.abspath(path))

DATASET_DIR = Path("./datasets")


def inspect_folder(image_dir, mask_dir, image_extensions):
    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    masks = sorted([
        p for p in mask_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    print(f"Images : {len(images)}")
    print(f"Masks  : {len(masks)}")

    if len(images) != len(masks):
        print("WARNING: Image and mask counts do not match!")
    else:
        print("✓ Image/mask counts match")

    print("\nFirst 5 images:")
    for p in images[:5]:
        print(" ", p.name)

    print("\nFirst 5 masks:")
    for p in masks[:5]:
        print(" ", p.name)

    if images:
        img = cv2.imread(images[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        print("\nFirst image:")
        print("  Name      :", images[0].name)
        print("  Size      :", img.size)
        print("  Mode      :", img.mode)
        print("  Format    :", img.format)

    if masks:
        mask = cv2.imread(masks[0])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = Image.fromarray(mask)
        print("\nFirst mask:")
        print("  Name      :", masks[0].name)
        print("  Size      :", mask.size)
        print("  Mode      :", mask.mode)
        print("  Format    :", mask.format)


# --------------------------------------------------
# Kvasir-SEG
# --------------------------------------------------

print("=" * 60)
print("KVASIR-SEG")
print("=" * 60)

kvasir_images = DATASET_DIR / "Kvasir-SEG" / "images"
kvasir_masks = DATASET_DIR / "Kvasir-SEG" / "masks"

inspect_folder(
    kvasir_images,
    kvasir_masks,
    {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
)


# --------------------------------------------------
# CVC-ClinicDB
# --------------------------------------------------

print("\n" + "=" * 60)
print("CVC-CLINICDB")
print("=" * 60)

cvc_images = DATASET_DIR / "CVC-ClinicDB" / "Original"
cvc_masks = DATASET_DIR / "CVC-ClinicDB" / "Ground Truth"

inspect_folder(
    cvc_images,
    cvc_masks,
    {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
)