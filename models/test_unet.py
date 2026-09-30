import torch

from unet import UNet


def main():

    model = UNet()

    x = torch.randn(
        2, 3, 352, 352
    )

    output = model(x)

    print("=" * 60)
    print("U-NET TEST")
    print("=" * 60)

    print("Input shape  :", x.shape)
    print("Output shape :", output.shape)
    print("Output dtype :", output.dtype)

    print("=" * 60)


if __name__ == "__main__":
    main()