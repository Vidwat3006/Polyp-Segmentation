from pathlib import Path
import sys

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from torch.utils.data import DataLoader

from preprocessing.dataset import (
    KvasirDataset,
    CVCClinicDBDataset
)


# ============================================================
# Paths
# ============================================================

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
# Configuration
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# Create datasets
# ============================================================

train_dataset = KvasirDataset(
    root_dir=KVASIR_ROOT,
    split_file=SPLIT_FILE,
    split="train"
)

val_dataset = KvasirDataset(
    root_dir=KVASIR_ROOT,
    split_file=SPLIT_FILE,
    split="validation"
)

test_dataset = CVCClinicDBDataset(
    root_dir=CVC_ROOT
)


# ============================================================
# Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DATALOADER INFORMATION")
    print("=" * 60)

    print("Training samples   :", len(train_dataset))
    print("Validation samples :", len(val_dataset))
    print("Test samples       :", len(test_dataset))

    print()

    print("Batch size         :", BATCH_SIZE)

    print(
        "Training batches   :",
        len(train_loader)
    )

    print(
        "Validation batches :",
        len(val_loader)
    )

    print(
        "Test batches       :",
        len(test_loader)
    )

    # --------------------------------------------------------
    # Get one training batch
    # --------------------------------------------------------

    images, masks = next(iter(train_loader))

    print("\n" + "=" * 60)
    print("TRAINING BATCH")
    print("=" * 60)

    print("Images shape :", images.shape)
    print("Images dtype :", images.dtype)

    print("Masks shape  :", masks.shape)
    print("Masks dtype  :", masks.dtype)

    print(
        "Mask values  :",
        masks.unique()
    )