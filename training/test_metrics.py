import torch

from losses import BCEDiceLoss

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score
)


def main():

    # Simulate model output
    logits = torch.randn(
        2, 1, 352, 352
    )

    # Simulate binary ground-truth masks
    targets = torch.randint(
        0,
        2,
        (2, 1, 352, 352)
    ).float()

    # -------------------------
    # Loss
    # -------------------------

    criterion = BCEDiceLoss()

    loss = criterion(
        logits,
        targets
    )

    # -------------------------
    # Metrics
    # -------------------------

    dice = dice_score(
        logits,
        targets
    )

    iou = iou_score(
        logits,
        targets
    )

    precision = precision_score(
        logits,
        targets
    )

    recall = recall_score(
        logits,
        targets
    )

    # -------------------------
    # Print results
    # -------------------------

    print("=" * 60)
    print("LOSS + METRICS TEST")
    print("=" * 60)

    print("Logits shape :", logits.shape)
    print("Target shape :", targets.shape)

    print()

    print("Loss      :", loss.item())
    print("Dice      :", dice)
    print("IoU       :", iou)
    print("Precision :", precision)
    print("Recall    :", recall)

    print("=" * 60)


if __name__ == "__main__":
    main()