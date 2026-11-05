
$ErrorActionPreference = "Stop"

# Project configuration
$DiffusersDir = "..\diffusers"
$DataDir = "data\lora_train"
$OutputDir = "outputs\lora_coco_test"
$ModelId = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Training script
$Script = Join-Path $DiffusersDir "examples\text_to_image\train_text_to_image_lora.py"

if (!(Test-Path $Script)) {
    throw "Diffusers training script not found."
}

if (!(Test-Path "$DataDir\metadata.jsonl")) {
    throw "Training dataset metadata not found."
}

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# RTX 4050 conservative LoRA test
accelerate launch --num_processes=1 --mixed_precision="fp16" $Script `
    --pretrained_model_name_or_path=$ModelId `
    --train_data_dir=$DataDir `
    --image_column="image" `
    --caption_column="text" `
    --resolution=512 `
    --center_crop `
    --train_batch_size=1 `
    --gradient_accumulation_steps=4 `
    --gradient_checkpointing `
    --learning_rate=1e-4 `
    --lr_scheduler="constant" `
    --lr_warmup_steps=0 `
    --max_train_steps=100 `
    --max_train_samples=1000 `
    --dataloader_num_workers=0 `
    --rank=4 `
    --checkpointing_steps=50 `
    --seed=42 `
    --mixed_precision="fp16" `
    --output_dir=$OutputDir `
    --report_to="tensorboard"