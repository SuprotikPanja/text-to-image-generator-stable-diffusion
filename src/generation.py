
import gc
import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from .config import settings
from .safety import validate_prompt, validate_dimensions


class TextToImageEngine:
    def __init__(self):
        self.pipe: Optional[StableDiffusionPipeline] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_loaded = False

    def load(self):
        if self.pipe is not None:
            return

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print("Loading Stable Diffusion:", settings.model_id)

        pipe = StableDiffusionPipeline.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            use_safetensors=True
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        pipe.enable_attention_slicing()

        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")

        # Load the optional LoRA adapter.
        if settings.lora_path:
            adapter_path = Path(settings.lora_path)

            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"LoRA adapter folder not found: {adapter_path}"
                )

            pipe.load_lora_weights(
                str(adapter_path),
                weight_name="pytorch_lora_weights.safetensors"
            )

            self.lora_loaded = True
            print("LoRA adapter loaded successfully.")

        self.pipe = pipe

    def generate(
        self,
        prompt,
        negative_prompt="",
        width=512,
        height=512,
        steps=20,
        guidance=7.0,
        seed=-1,
        use_lora=False,
        lora_strength=0.75
    ):
        prompt = validate_prompt(prompt)

        width, height = validate_dimensions(
            width, height, settings.max_image_side
        )

        steps = max(10, min(int(steps), 35))
        guidance = max(1.0, min(float(guidance), 12.0))
        lora_strength = max(0.0, min(float(lora_strength), 1.0))

        self.load()

        if use_lora and not self.lora_loaded:
            raise RuntimeError(
                "LoRA is enabled, but no adapter is configured. "
                "Set LORA_PATH in your environment configuration."
            )

        if int(seed) < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        generator = torch.Generator(
            device="cuda" if self.device == "cuda" else "cpu"
        ).manual_seed(int(seed))

        call_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator
        }

        if self.lora_loaded:
            call_args["cross_attention_kwargs"] = {
                "scale": lora_strength if use_lora else 0.0
            }

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        with torch.inference_mode():
            image = self.pipe(**call_args).images[0]

        elapsed = round(time.perf_counter() - start, 2)

        meta = {
            "model": settings.model_id,
            "lora_enabled": bool(use_lora and self.lora_loaded),
            "lora_strength": lora_strength if use_lora else 0.0,
            "seed": int(seed),
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "device": self.device,
            "generation_seconds": elapsed
        }

        if self.device == "cuda":
            meta["gpu_peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )
            torch.cuda.empty_cache()

        gc.collect()

        return image, meta


engine = TextToImageEngine()
