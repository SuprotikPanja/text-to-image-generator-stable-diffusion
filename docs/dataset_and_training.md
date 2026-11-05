# Dataset and task execution

## Dataset recommendation
Use COCO Captions as the primary dataset because it pairs everyday images with natural-language captions. Begin with a curated 500–2,000 pair subset to validate the process. Oxford-102 Flowers is optional domain-specific data, but it has flower labels/images rather than rich captions; generate and manually review captions before text-to-image training. Ask ElevanceSkills to approve the dataset choice first. Check original image licenses and the dataset card; do not assume all third-party images permit redistribution/commercial use. Keep raw data out of GitHub.

Create an input JSONL manifest, one row per image: `{"image":"images/00001.jpg","text":"A red bicycle parked beside a brick wall."}`. Paths can be relative to the manifest. Run `python scripts/prepare_caption_jsonl.py --input data/captions.jsonl --output-dir data/train` to copy validated images and create the Hugging Face imagefolder `metadata.jsonl` manifest.

## Task 4 — dataset analysis
`python -m src.data.coco_analysis --jsonl data/captions.jsonl --limit 2000`
The script reports caption word counts, image resolutions and bad records. It does not download COCO.

## Task 3 — text encoder
Run the app and use the Text preprocessing tab. Stable Diffusion 1.5 expects its checkpoint-matched CLIP tokenizer/encoder; BERT is not a drop-in replacement.

## Tasks 2 and 5 — GAN
Run baseline and attention variants:
`python -m src.gan.train_shapes --epochs 20 --batch 64 --out outputs/cgan.pt`
`python -m src.gan.train_shapes --epochs 20 --batch 64 --attention --out outputs/attention_cgan.pt`
This uses a generated shapes dataset (square/circle/triangle), so no external dataset is needed. Compare samples and losses. This is an educational low-resolution GAN experiment, not a rival to Stable Diffusion.

## Task 1 — LoRA
Use the official Diffusers LoRA training example: https://huggingface.co/docs/diffusers/en/training/lora . Clone the Diffusers repo beside this project and follow its current training requirements. Prepare an imagefolder dataset in `data/train` as described above. Use batch 1, 512px, mixed precision, gradient checkpointing and a small step count initially. 6 GB VRAM may still be insufficient for some software/driver combinations; use a cloud GPU if OOM persists. Save the LoRA adapter and load it into the app via `LORA_PATH`.

Do not claim to have fine-tuned if you only used pretrained inference. Compare base vs LoRA using the same prompts/seeds.
