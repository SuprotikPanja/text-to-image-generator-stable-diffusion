# Mapping to the six tasks

| Task | Code/evidence |
|---|---|
| 1 Fine-tune pretrained model | Diffusers LoRA adapter; training logs and base-vs-LoRA comparison |
| 2 Conditional GAN | `src/gan/models.py` and `train_shapes.py`; generated samples |
| 3 Text preprocessing | `src/text.py`; token IDs and CLIP embedding shape |
| 4 Dataset analysis | `src/data/coco_analysis.py`; dataset card, statistics and limitations |
| 5 Attention GAN | `SelfAttention2d`; baseline vs attention comparison |
| 6 End-to-end pipeline | `app.py`; public demo, README, screenshots and tests |

The GAN modules are separate model experiments. Stable Diffusion remains the main text-to-image generator. Explain this honestly in the report.
