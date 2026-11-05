
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ADAPTER_DIR = Path("outputs/lora_coco_test")
OUTPUT_DIR = Path("outputs/lora_evaluation")

ADAPTER_FILE = ADAPTER_DIR / "pytorch_lora_weights.safetensors"

if not ADAPTER_FILE.exists():
    raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_FILE}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

prompts = [
    "A black motorcycle parked in front of a garage, realistic photography",
    "A giraffe running across a green grass field, wildlife photography",
    "A room with blue walls and a white sink and door, realistic photography",
]

print("Loading Stable Diffusion 1.5...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

# Memory optimizations for RTX 4050 6GB
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing("max")

pipe.set_progress_bar_config(disable=False)

comparison_images = []

for index, prompt in enumerate(prompts, start=1):
    print(f"\nPrompt {index}: {prompt}")

    # Baseline generation
    baseline_generator = torch.Generator(
        device="cuda"
    ).manual_seed(42)

    baseline = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=baseline_generator,
    ).images[0]

    baseline_path = OUTPUT_DIR / f"baseline_{index}.png"
    baseline.save(baseline_path)

    # Load trained LoRA adapter
    if index == 1:
        pipe.load_lora_weights(
            str(ADAPTER_DIR),
            weight_name="pytorch_lora_weights.safetensors",
        )
        print("LoRA adapter loaded successfully.")

    # Fine-tuned generation with identical seed
    lora_generator = torch.Generator(
        device="cuda"
    ).manual_seed(42)

    tuned = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=lora_generator,
    ).images[0]

    tuned_path = OUTPUT_DIR / f"lora_{index}.png"
    tuned.save(tuned_path)

    # Side-by-side comparison
    panel = Image.new("RGB", (1024, 560), "white")
    panel.paste(baseline, (0, 48))
    panel.paste(tuned, (512, 48))

    draw = ImageDraw.Draw(panel)
    draw.text((200, 15), "BASELINE SD 1.5", fill="black")
    draw.text((700, 15), "SD 1.5 + LoRA", fill="black")

    comparison_path = OUTPUT_DIR / f"comparison_{index}.png"
    panel.save(comparison_path)

    comparison_images.append(comparison_path)

    print("Saved:", comparison_path)

print("\nEvaluation completed.")
print("Results folder:", OUTPUT_DIR.resolve())
