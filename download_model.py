"""Downloads and caches the CLIP model locally inside this project folder."""
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)
os.environ["HF_HOME"] = str(MODEL_DIR)

from transformers import CLIPModel, CLIPProcessor

print("Downloading openai/clip-vit-base-patch32 (~340 MB)…")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_DIR)
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_DIR)
print(f"Done. Model saved to: {MODEL_DIR}")
