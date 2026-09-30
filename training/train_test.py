import sys
from pathlib import Path

import torch
import torch.optim as optim

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.unet import UNet
from training.dataloader import train_loader
from training.losses import BCEDiceLoss


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("U-NET TRAINING SANITY TEST")
    print("=" * 60)

    print("Device :", device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = UNet()

    model = model.to(device)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = BCEDiceLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # --------------------------------------------------
    # Take only a few batches
    # --------------------------------------------------

    model.train()

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            masks
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(
            f"Batch {batch_idx + 1} | "
            f"Loss: {loss.item():.4f}"
        )

        # Only test first 5 batches
        if batch_idx == 4:
            break

    print("=" * 60)
    print("TRAINING SANITY TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()