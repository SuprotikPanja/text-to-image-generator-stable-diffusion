
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .models import ConditionalGenerator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="outputs/cgan_shapes.pt"
    )
    parser.add_argument(
        "--out",
        default="outputs/gan_samples.png"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # Reproducible random generation
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load trained checkpoint
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    # Recreate the correct generator architecture
    model = ConditionalGenerator(
        attention=checkpoint.get("attention", False)
    )

    model.load_state_dict(checkpoint["generator"])
    model.eval()

    # Generate the same labels and random noise for comparison
    labels = torch.arange(args.count) % 3
    noise = torch.randn(args.count, 64)

    with torch.no_grad():
        images = model(noise, labels)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        images,
        str(output_path),
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    print("Generated samples:", output_path)


if __name__ == "__main__":
    main()
