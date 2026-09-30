from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 352


# ============================================================
# Training Transform
# ============================================================

train_transform = A.Compose([
    A.Resize(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE
    ),

    A.HorizontalFlip(
        p=0.5
    ),

    A.VerticalFlip(
        p=0.2
    ),

    A.Rotate(
        limit=15,
        p=0.5,
        border_mode=cv2.BORDER_REFLECT_101
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


# ============================================================
# Validation / Test Transform
# ============================================================

eval_transform = A.Compose([
    A.Resize(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


# ============================================================
# Base Dataset
# ============================================================

class PolypDataset(Dataset):

    def __init__(
        self,
        image_paths,
        mask_paths,
        transform=None
    ):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        assert len(self.image_paths) == len(
            self.mask_paths
        ), "Number of images and masks must match."

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # OpenCV: BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(
                f"Could not read mask: {mask_path}"
            )

        # ----------------------------------------------------
        # Convert mask to binary
        # ----------------------------------------------------

        mask = (mask > 127).astype(np.float32)

        # ----------------------------------------------------
        # Apply transforms
        # ----------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # ----------------------------------------------------
        # Ensure mask shape = [1, H, W]
        # ----------------------------------------------------

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # ----------------------------------------------------
        # Convert mask to float
        # ----------------------------------------------------

        mask = mask.float()

        return image, mask


# ============================================================
# Kvasir Dataset
# ============================================================

class KvasirDataset(PolypDataset):

    def __init__(
        self,
        root_dir,
        split_file,
        split="train"
    ):

        root_dir = Path(root_dir)

        image_dir = root_dir / "images"
        mask_dir = root_dir / "masks"

        # ----------------------------------------------------
        # Read split
        # ----------------------------------------------------

        with open(split_file, "r") as f:
            split_data = json.load(f)

        filenames = split_data[split]

        image_paths = []
        mask_paths = []

        for filename in filenames:

            image_path = image_dir / filename
            mask_path = mask_dir / filename

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            image_paths.append(image_path)
            mask_paths.append(mask_path)

        # ----------------------------------------------------
        # Select transform
        # ----------------------------------------------------

        if split == "train":
            transform = train_transform
        else:
            transform = eval_transform

        super().__init__(
            image_paths,
            mask_paths,
            transform
        )


# ============================================================
# CVC-ClinicDB Dataset
# ============================================================

class CVCClinicDBDataset(PolypDataset):

    def __init__(
        self,
        root_dir
    ):

        root_dir = Path(root_dir)

        image_dir = root_dir / "Original"
        mask_dir = root_dir / "Ground Truth"

        image_paths = sorted(
            image_dir.glob("*.tif")
        )

        mask_paths = []

        for image_path in image_paths:

            mask_path = mask_dir / image_path.name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            mask_paths.append(mask_path)

        super().__init__(
            image_paths,
            mask_paths,
            eval_transform
        )