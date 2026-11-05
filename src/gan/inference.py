
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from .models import ConditionalGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS = {
    "Baseline CGAN": PROJECT_ROOT / "outputs/experiments/baseline/cgan_baseline.pt",
    "Attention CGAN": PROJECT_ROOT / "outputs/experiments/attention/cgan_attention_shapes.pt",
}

SHAPE_LABELS = {
    "Square": 0,
    "Circle": 1,
    "Triangle": 2,
}


def generate_gan_samples(
    model_name="Baseline CGAN",
    shape_name="Square",
    count=8,
    seed=42
):
    if model_name not in CHECKPOINTS:
        raise ValueError("Unknown GAN model selected.")

    if shape_name not in SHAPE_LABELS:
        raise ValueError("Unknown shape selected.")

    count = max(1, min(int(count), 16))
    seed = int(seed)

    checkpoint_path = CHECKPOINTS[model_name]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = ConditionalGenerator(
        num_classes=checkpoint.get("num_classes", 3),
        latent=checkpoint.get("latent", 64),
        attention=checkpoint.get("attention", False)
    )

    model.load_state_dict(checkpoint["generator"])
    model.to(device)
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    labels = torch.full(
        (count,),
        SHAPE_LABELS[shape_name],
        dtype=torch.long,
        device=device
    )

    noise = torch.randn(
        count,
        checkpoint.get("latent", 64),
        generator=generator,
        device=device
    )

    with torch.inference_mode():
        images = model(noise, labels)

    images = (images.cpu() + 1) / 2
    images = images.clamp(0, 1)

    grid = make_grid(
        images,
        nrow=min(4, count),
        padding=4
    )

    result = to_pil_image(grid)

    metadata = {
        "model": model_name,
        "shape": shape_name,
        "count": count,
        "seed": seed,
        "device": str(device),
        "image_size": "64x64",
        "attention_enabled": checkpoint.get("attention", False)
    }

    return result, metadata
