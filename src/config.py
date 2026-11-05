import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    lora_path: str = os.getenv("LORA_PATH", "").strip()
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "7860"))
    max_image_side: int = int(os.getenv("MAX_IMAGE_SIDE", "768"))
    default_steps: int = int(os.getenv("DEFAULT_STEPS", "20"))

settings = Settings()
