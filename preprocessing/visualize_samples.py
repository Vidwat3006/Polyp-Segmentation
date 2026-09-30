# from pathlib import Path
# from PIL import Image
# import matplotlib.pyplot as plt


# DATASET_DIR = Path("./datasets")


# def show_samples(image_dir, mask_dir, image_names, title):
#     fig, axes = plt.subplots(
#         len(image_names),
#         3,
#         figsize=(12, 4 * len(image_names))
#     )

#     if len(image_names) == 1:
#         axes = [axes]

#     for row, name in enumerate(image_names):

#         image = Image.open(image_dir / name).convert("RGB")
#         mask = Image.open(mask_dir / name).convert("RGB")

#         axes[row][0].imshow(image)
#         axes[row][0].set_title("Original Image")
#         axes[row][0].axis("off")

#         axes[row][1].imshow(mask)
#         axes[row][1].set_title("Ground Truth Mask")
#         axes[row][1].axis("off")

#         axes[row][2].imshow(image)
#         axes[row][2].imshow(mask, alpha=0.4)
#         axes[row][2].set_title("Overlay")
#         axes[row][2].axis("off")

#     fig.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()


# # --------------------------------------------------
# # Kvasir
# # --------------------------------------------------

# kvasir_images = DATASET_DIR / "Kvasir-SEG" / "images"
# kvasir_masks = DATASET_DIR / "Kvasir-SEG" / "masks"

# kvasir_names = [
#     "cju0qkwl35piu0993l0dewei2.jpg",
#     "cju0qoxqj9q6s0835b43399p4.jpg",
#     "cju0qx73cjw570799j4n5cjze.jpg",
# ]

# show_samples(
#     kvasir_images,
#     kvasir_masks,
#     kvasir_names,
#     "Kvasir-SEG Samples"
# )


# # --------------------------------------------------
# # CVC
# # --------------------------------------------------

# cvc_images = DATASET_DIR / "CVC-ClinicDB" / "Original"
# cvc_masks = DATASET_DIR / "CVC-ClinicDB" / "Ground Truth"

# cvc_names = [
#     "1.tif",
#     "10.tif",
#     "100.tif",
# ]

# show_samples(
#     cvc_images,
#     cvc_masks,
#     cvc_names,
#     "CVC-ClinicDB Samples"
# )



from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


DATASET_DIR = Path("./datasets")


def load_image(path):
    """
    Load JPG/PNG using PIL and TIFF using OpenCV.
    Returns RGB numpy array.
    """

    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {path}")

        # OpenCV loads BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    else:
        return np.array(Image.open(path).convert("RGB"))


def show_samples(image_dir, mask_dir, image_names, title):

    fig, axes = plt.subplots(
        len(image_names),
        3,
        figsize=(12, 4 * len(image_names))
    )

    if len(image_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, name in enumerate(image_names):

        image_path = image_dir / name
        mask_path = mask_dir / name

        image = load_image(image_path)
        mask = load_image(mask_path)

        axes[row][0].imshow(image)
        axes[row][0].set_title("Original Image")
        axes[row][0].axis("off")

        axes[row][1].imshow(mask)
        axes[row][1].set_title("Ground Truth Mask")
        axes[row][1].axis("off")

        axes[row][2].imshow(image)

        # Convert mask to grayscale for overlay
        mask_gray = cv2.cvtColor(
            mask,
            cv2.COLOR_RGB2GRAY
        )

        # Show only non-zero mask pixels
        axes[row][2].imshow(
            mask_gray,
            alpha=0.4
        )

        axes[row][2].set_title("Overlay")
        axes[row][2].axis("off")

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


# ============================================================
# KVASIR-SEG
# ============================================================

kvasir_images = DATASET_DIR / "Kvasir-SEG" / "images"
kvasir_masks = DATASET_DIR / "Kvasir-SEG" / "masks"

kvasir_names = [
    "cju0qkwl35piu0993l0dewei2.jpg",
    "cju0qoxqj9q6s0835b43399p4.jpg",
    "cju0qx73cjw570799j4n5cjze.jpg",
]

show_samples(
    kvasir_images,
    kvasir_masks,
    kvasir_names,
    "Kvasir-SEG Samples"
)


# ============================================================
# CVC-CLINICDB
# ============================================================

cvc_images = DATASET_DIR / "CVC-ClinicDB" / "Original"
cvc_masks = DATASET_DIR / "CVC-ClinicDB" / "Ground Truth"

cvc_names = [
    "1.tif",
    "10.tif",
    "100.tif",
]

show_samples(
    cvc_images,
    cvc_masks,
    cvc_names,
    "CVC-ClinicDB Samples"
)