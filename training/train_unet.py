import sys
from pathlib import Path
import time

import torch
import torch.optim as optim


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Imports
# --------------------------------------------------

from models.unet import UNet
from training.dataloader import train_loader, val_loader
from training.losses import BCEDiceLoss

from evaluation.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_EPOCHS = 1

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif hasattr(torch, "xpu") and torch.xpu.is_available():

    DEVICE = torch.device("xpu")

else:

    DEVICE = torch.device("cpu")


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            masks
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss / len(loader)
    )

    return epoch_loss


# --------------------------------------------------
# Validation function
# --------------------------------------------------

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    dice_total = 0.0
    iou_total = 0.0
    precision_total = 0.0
    recall_total = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(images)

            # Loss
            loss = criterion(
                outputs,
                masks
            )

            running_loss += loss.item()

            # Metrics
            dice_total += dice_score(
                outputs,
                masks
            )

            iou_total += iou_score(
                outputs,
                masks
            )

            precision_total += precision_score(
                outputs,
                masks
            )

            recall_total += recall_score(
                outputs,
                masks
            )

    num_batches = len(loader)

    return (
        running_loss / num_batches,
        dice_total / num_batches,
        iou_total / num_batches,
        precision_total / num_batches,
        recall_total / num_batches
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("U-NET TRAINING")
    print("=" * 70)

    print("Device        :", DEVICE)
    print("Epochs        :", NUM_EPOCHS)
    print("Learning rate :", LEARNING_RATE)

    print()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = UNet()

    model = model.to(DEVICE)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = BCEDiceLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------
    # Best validation Dice
    # --------------------------------------------------

    best_dice = 0.0

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        start_time = time.time()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        # Training
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        # Validation
        (
            val_loss,
            val_dice,
            val_iou,
            val_precision,
            val_recall
        ) = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        elapsed = (
            time.time() - start_time
        )

        print(
            f"Train Loss   : {train_loss:.4f}"
        )

        print(
            f"Val Loss     : {val_loss:.4f}"
        )

        print(
            f"Val Dice     : {val_dice:.4f}"
        )

        print(
            f"Val IoU      : {val_iou:.4f}"
        )

        print(
            f"Val Precision: {val_precision:.4f}"
        )

        print(
            f"Val Recall   : {val_recall:.4f}"
        )

        print(
            f"Time         : {elapsed:.2f} sec"
        )

        print("-" * 70)

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            checkpoint_path = (
                CHECKPOINT_DIR /
                "unet_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                checkpoint_path
            )

            print(
                f"Saved best model → {checkpoint_path}"
            )

    print()

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best Validation Dice: {best_dice:.4f}"
    )


if __name__ == "__main__":
    main()