# Text-to-Image Studio — ElevanceSkills Internship Project

One repository for the six related tasks. The supplied Stable Diffusion demo remains the baseline.

## Model and dataset decisions
- Main generator: pretrained Stable Diffusion 1.5 + optional LoRA adapter.
- Main dataset: COCO Captions, using a small curated subset of image/caption pairs first.
- Oxford-102 Flowers is optional domain-specific work only: it has flower labels/images, not rich natural-language captions. Create and review captions before text-to-image fine-tuning.
- GAN tasks: a separate label-conditioned shapes GAN, plus a self-attention variant. GANs are not the same architecture as Stable Diffusion.
- 6 GB VRAM: use 512px, batch size 1, mixed precision and LoRA. Do not train a diffusion model from scratch.
- Public free GPU hosting is not guaranteed; CPU inference can be very slow. Check current provider hardware, quota and pricing.

Read `docs/beginner_setup.md`, then `docs/dataset_and_training.md`, then `docs/deployment.md`.
