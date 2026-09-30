from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path("./datasets")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}


def get_files(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def analyze_dataset(name, image_dir, mask_dir):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    images = get_files(image_dir)
    masks = get_files(mask_dir)

    print("Images:", len(images))
    print("Masks :", len(masks))

    image_names = {p.name for p in images}
    mask_names = {p.name for p in masks}

    missing_masks = image_names - mask_names
    missing_images = mask_names - image_names

    print("\nMissing masks:", len(missing_masks))
    print("Missing images:", len(missing_images))

    if missing_masks:
        print("Examples:", list(missing_masks)[:10])

    if missing_images:
        print("Examples:", list(missing_images)[:10])

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    dimension_mismatches = []

    empty_masks = []
    full_masks = []

    foreground_percentages = []

    low_value_foreground = []

    for image_path in images:

        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            continue

        # Read image
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        # Read mask as grayscale
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print("Could not read image:", image_path)
            continue

        if mask is None:
            print("Could not read mask:", mask_path)
            continue

        # --------------------------------------------------
        # Dimension check
        # --------------------------------------------------

        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]

        if (
            image_height != mask_height
            or image_width != mask_width
        ):
            dimension_mismatches.append(
                (
                    image_path.name,
                    image.shape,
                    mask.shape
                )
            )

        # --------------------------------------------------
        # Binary mask using threshold 127
        # --------------------------------------------------

        binary_mask = mask > 127

        foreground_pixels = np.count_nonzero(
            binary_mask
        )

        total_pixels = binary_mask.size

        foreground_percentage = (
            foreground_pixels /
            total_pixels *
            100
        )

        foreground_percentages.append(
            foreground_percentage
        )

        # --------------------------------------------------
        # Empty/full mask
        # --------------------------------------------------

        if foreground_pixels == 0:
            empty_masks.append(image_path.name)

        if foreground_pixels == total_pixels:
            full_masks.append(image_path.name)

        # --------------------------------------------------
        # Check for low-valued pixels
        # --------------------------------------------------

        low_values = (
            (mask > 0) &
            (mask <= 127)
        )

        if np.any(low_values):

            low_value_foreground.append(
                (
                    image_path.name,
                    int(np.count_nonzero(low_values))
                )
            )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print("\nDimension mismatches:")
    print(len(dimension_mismatches))

    if dimension_mismatches:
        for item in dimension_mismatches[:10]:
            print(item)

    print("\nEmpty masks:")
    print(len(empty_masks))

    if empty_masks:
        print(empty_masks[:10])

    print("\nFull masks:")
    print(len(full_masks))

    if full_masks:
        print(full_masks[:10])

    # ------------------------------------------------------
    # Foreground statistics
    # ------------------------------------------------------

    if foreground_percentages:

        percentages = np.array(
            foreground_percentages
        )

        print("\nForeground percentage:")
        print(
            f"Minimum : {percentages.min():.2f}%"
        )
        print(
            f"Maximum : {percentages.max():.2f}%"
        )
        print(
            f"Mean    : {percentages.mean():.2f}%"
        )
        print(
            f"Median  : {np.median(percentages):.2f}%"
        )

    print(
        "\nImages containing low-valued "
        "mask pixels (1-127):",
        len(low_value_foreground)
    )

    if low_value_foreground:
        print(
            "Examples:",
            low_value_foreground[:10]
        )


# ==========================================================
# Kvasir-SEG
# ==========================================================

analyze_dataset(
    "KVASIR-SEG",
    DATASET_DIR / "Kvasir-SEG" / "images",
    DATASET_DIR / "Kvasir-SEG" / "masks"
)


# ==========================================================
# CVC-ClinicDB
# ==========================================================

analyze_dataset(
    "CVC-CLINICDB",
    DATASET_DIR / "CVC-ClinicDB" / "Original",
    DATASET_DIR / "CVC-ClinicDB" / "Ground Truth"
)