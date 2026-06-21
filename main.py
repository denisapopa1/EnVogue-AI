import io
import os
import shutil
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI Wardrobe Stylist", version="2.0")

BASE_DIR = Path(__file__).parent
BASE_IMAGES = BASE_DIR / "BaseImages"   # BaseImages/{user_id}/Category/Sub/img.jpg
MODEL_DIR = BASE_DIR / "model"

# Create all required base folders on startup.
BASE_IMAGES.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Generated outfits live at BaseImages/{user_id}/generated/*.jpg
app.mount("/images", StaticFiles(directory=str(BASE_IMAGES)), name="images")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CATEGORY_NAMES = {"Accessories", "Bottoms", "Dress", "Outerwear", "Shoes", "Tops"}

# ---------------------------------------------------------------------------
# Load CLIP  (openai/clip-vit-base-patch32 — ~340 MB, CPU-friendly)
# Run download_model.py once to cache locally.
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_DIR))
os.environ.setdefault("HF_HOME", str(MODEL_DIR))

print("Loading CLIP model…")
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_DIR)
_clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_DIR)
_clip_model.eval()
print("CLIP model ready.")

# In-memory embedding cache: wardrobe_path_str → {img_path_str: np.ndarray}
_emb_cache: Dict[str, Dict[str, np.ndarray]] = {}

# ---------------------------------------------------------------------------
# Filter enums — rendered as dropdowns in Swagger UI
# ---------------------------------------------------------------------------

class Style(str, Enum):
    casual  = "casual"
    formal  = "formal"
    elegant = "elegant"
    sporty  = "sporty"

class Season(str, Enum):
    winter = "winter"
    spring = "spring"
    summer = "summer"
    autumn = "autumn"

class Weather(str, Enum):
    sunny  = "sunny"
    cloudy = "cloudy"
    rainy  = "rainy"
    cold   = "cold"

# ---------------------------------------------------------------------------
# Text prompts for CLIP scoring
# ---------------------------------------------------------------------------
STYLE_TEXT = {
    "casual":  "casual everyday outfit, relaxed streetwear, comfortable clothing",
    "formal":  "formal professional business attire, office wear",
    "elegant": "elegant sophisticated evening wear, chic classy luxury fashion",
    "sporty":  "sporty athletic activewear, gym workout outfit",
}
SEASON_TEXT = {
    "winter": "warm winter clothing, heavy coats and sweaters for cold weather",
    "spring": "light spring clothing, mild weather layers",
    "summer": "light summer outfit, shorts and t-shirts for hot weather",
    "autumn": "cozy autumn outfit, layered clothing for cool fall weather",
}
WEATHER_TEXT = {
    "sunny":  "light fabrics for bright sunny warm weather",
    "cloudy": "medium layers for overcast cloudy day",
    "rainy":  "waterproof clothing for rainy day, rain boots and coat",
    "cold":   "warm heavy clothing and insulating layers for freezing cold weather",
}
OUTFIT_NAMES = {
    "casual":  ["Weekend Casual", "Everyday Relaxed", "Easy Day Out"],
    "formal":  ["Business Professional", "Office Ready", "Smart Formal"],
    "elegant": ["Evening Elegance", "Sophisticated Chic", "Classic Glamour"],
    "sporty":  ["Active Day", "Sport & Go", "Athletic Edge"],
}


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def _image_embedding(path: Path) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
        inputs = _clip_proc(images=img, return_tensors="pt")
        with torch.no_grad():
            vo = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
            feat = _clip_model.visual_projection(vo.pooler_output)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().numpy()
    except Exception:
        return None


def _text_embedding(text: str) -> np.ndarray:
    inputs = _clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        to = _clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        feat = _clip_model.text_projection(to.pooler_output)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Wardrobe helpers
# ---------------------------------------------------------------------------

def get_wardrobe_path(user_id: str) -> Path:
    user_path = BASE_IMAGES / user_id
    if user_path.exists():
        return user_path
    raise HTTPException(404, f"No wardrobe for '{user_id}'. Upload a zip first.")


def find_category_root(base: Path) -> Path:
    """Walk an extracted tree and return the dir that directly contains category folders."""
    for root, dirs, _ in os.walk(base):
        if set(dirs) & CATEGORY_NAMES:
            return Path(root)
    return base


def scan_wardrobe(path: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Returns {category: {subcategory: [image_paths]}}, skipping the generated/ folder."""
    inventory: Dict[str, Dict[str, List[Path]]] = {}
    for cat_dir in sorted(path.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "generated":
            continue
        subcats: Dict[str, List[Path]] = {}
        for child in sorted(cat_dir.iterdir()):
            if child.is_dir():
                imgs = [p for p in child.iterdir() if p.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    subcats[child.name] = imgs
            elif child.suffix.lower() in IMAGE_EXTS:
                subcats.setdefault("_items", []).append(child)
        if subcats:
            inventory[cat_dir.name] = subcats
    return inventory


def ensure_embeddings(wpath: Path, inventory: Dict[str, Dict[str, List[Path]]]):
    key = str(wpath)
    if key not in _emb_cache:
        _emb_cache[key] = {}
    cache = _emb_cache[key]
    new_count = 0
    for subcats in inventory.values():
        for imgs in subcats.values():
            for img_path in imgs:
                if str(img_path) not in cache:
                    emb = _image_embedding(img_path)
                    if emb is not None:
                        cache[str(img_path)] = emb
                        new_count += 1
    if new_count:
        print(f"Computed {new_count} new embeddings for {wpath.name}")


def rank_by_clip(
    subcats: Dict[str, List[Path]], wpath: Path, text_emb: np.ndarray, top_n: int = 4
) -> List[Tuple[str, Path, float]]:
    cache = _emb_cache.get(str(wpath), {})
    scored = [
        (subcat, img, _cosine(cache[str(img)], text_emb))
        for subcat, imgs in subcats.items()
        for img in imgs
        if str(img) in cache
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Outfit generation
# ---------------------------------------------------------------------------

def build_pool(
    inventory: Dict[str, Dict[str, List[Path]]],
    wpath: Path,
    style: str, season: str, weather: str,
    top_n: int = 4,
) -> Dict[str, List[Tuple[str, Path, float]]]:
    text = f"{STYLE_TEXT[style]}. {SEASON_TEXT[season]}. {WEATHER_TEXT[weather]}"
    text_emb = _text_embedding(text)
    return {
        cat: ranked
        for cat, subcats in inventory.items()
        if (ranked := rank_by_clip(subcats, wpath, text_emb, top_n))
    }


def _pick(pool: Dict, category: str, idx: int) -> Optional[dict]:
    items = pool.get(category)
    if not items:
        return None
    subcat, img_path, _ = items[idx % len(items)]
    return {"label": f"{category}/{subcat}", "role": category.lower().rstrip("s"), "image": img_path}


def _pick_preferred_subcategory(
    pool: Dict,
    category: str,
    preferred_subcategories: List[str],
    idx: int,
) -> Optional[dict]:
    items = pool.get(category)
    if not items:
        return None

    preferred = [
        (subcat, img_path, score)
        for subcat, img_path, score in items
        if any(subcat.lower() == preferred.lower() for preferred in preferred_subcategories)
    ]
    selected = preferred[idx % len(preferred)] if preferred else items[idx % len(items)]
    subcat, img_path, _ = selected
    return {"label": f"{category}/{subcat}", "role": category.lower().rstrip("s"), "image": img_path}


def make_suggestions(
    pool: Dict, style: str, season: str, weather: str, n: int = 3
) -> List[dict]:
    style_value = style.value if isinstance(style, Enum) else str(style)
    season_value = season.value if isinstance(season, Enum) else str(season)
    weather_value = weather.value if isinstance(weather, Enum) else str(weather)

    need_outerwear = season_value in ("winter", "autumn") or weather_value in ("cold", "rainy")
    suggestions = []
    for i in range(n):
        items = []
        use_dress = "Dress" in pool and style_value in ("elegant", "formal")

        if use_dress:
            item = _pick(pool, "Dress", i)
            if item:
                item["role"] = "dress"
                items.append(item)

        if not items:
            top = _pick(pool, "Tops", i)
            if top:
                top["role"] = "top"
                items.append(top)
            bottom = _pick(pool, "Bottoms", i)
            if bottom:
                bottom["role"] = "bottom"
                items.append(bottom)

        shoes = _pick(pool, "Shoes", i)
        if shoes:
            shoes["role"] = "shoes"
            items.append(shoes)

        if need_outerwear:
            outer = _pick(pool, "Outerwear", i)
            if outer:
                outer["role"] = "outerwear"
                items.append(outer)

        should_add_accessory = (
            style_value in ("elegant", "formal")
            or weather_value == "sunny"
            or i == 2
        )
        if should_add_accessory:
            preferred_accessories = ["Hat"] if weather_value == "sunny" else []
            if weather_value in ("cold", "rainy"):
                preferred_accessories = ["Scarf"]
            acc = _pick_preferred_subcategory(pool, "Accessories", preferred_accessories, i)
            if acc:
                acc["role"] = "accessory"
                items.append(acc)

        if items:
            names = OUTFIT_NAMES[style_value]
            suggestions.append({
                "name": names[i % len(names)],
                "description": f"A {style_value} outfit for {season_value}, {weather_value} weather.",
                "items": items,
            })
    return suggestions[:n]


# ---------------------------------------------------------------------------
# Collage
# ---------------------------------------------------------------------------

def create_collage(items: list, out_path: Path) -> bool:
    images: List[Image.Image] = []
    for item in items:
        p = item.get("image")
        if p:
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((280, 370))
                images.append(img)
            except Exception:
                pass
    if not images:
        return False

    cols = min(len(images), 4)
    rows = (len(images) + cols - 1) // cols
    pad, cw, ch = 8, 280, 370
    canvas = Image.new(
        "RGB",
        (cols * cw + (cols + 1) * pad, rows * ch + (rows + 1) * pad),
        (248, 248, 248),
    )
    for idx, img in enumerate(images):
        x = (idx % cols) * cw + (idx % cols + 1) * pad + (cw - img.width) // 2
        y = (idx // cols) * ch + (idx // cols + 1) * pad + (ch - img.height) // 2
        canvas.paste(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=85)
    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/wardrobe/{user_id}")
async def upload_wardrobe(
    user_id: str,
    file: UploadFile = File(...),
    replace: bool = Query(True, description="Replace the existing wardrobe for this user."),
):
    """
    Upload a wardrobe zip for a user.
    Expected zip layout: Category/Subcategory/image.jpg (any leading folders are stripped).
    Existing wardrobes are replaced by default so EnVogue can resync changed closets.
    """
    user_path = BASE_IMAGES / user_id
    if user_path.exists() and not replace:
        return {"message": "Wardrobe already exists, skipping extraction.", "user_id": user_id}
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip files are accepted.")

    tmp_dir = BASE_DIR / "tmp_extract"
    tmp_dir.mkdir(exist_ok=True)
    try:
        data = await file.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(tmp_dir)
        cat_root = find_category_root(tmp_dir)
        if user_path.exists():
            shutil.rmtree(user_path)
        shutil.copytree(cat_root, user_path)
        _emb_cache.pop(str(user_path), None)
        return {"message": "Wardrobe uploaded and ready.", "user_id": user_id,
                "path": f"BaseImages/{user_id}"}
    except Exception as e:
        raise HTTPException(400, f"Failed to process zip: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/outfit/{user_id}")
def generate_outfit(
    user_id: str,
    style: Style     = Style.casual,
    season: Season   = Season.spring,
    weather: Weather = Weather.sunny,
    count: int = Query(3, ge=1, le=12),
):
    """
    Generate outfit suggestions scored by CLIP image-text similarity.
    Collage images are saved to BaseImages/{user_id}/generated/ and served as static files.
    """
    style_value = style.value
    season_value = season.value
    weather_value = weather.value
    params = {"style": style_value, "season": season_value, "weather": weather_value, "count": count}

    wpath = get_wardrobe_path(user_id)
    inventory = scan_wardrobe(wpath)
    if not inventory:
        raise HTTPException(404, "Wardrobe is empty.")

    ensure_embeddings(wpath, inventory)
    pool = build_pool(inventory, wpath, style_value, season_value, weather_value, top_n=max(4, count))
    suggestions_raw = make_suggestions(pool, style_value, season_value, weather_value, n=count)

    out_dir = BASE_IMAGES / user_id / "generated"
    result = []
    for i, sug in enumerate(suggestions_raw):
        fname = f"outfit_{style_value}_{season_value}_{weather_value}_{i + 1}.jpg"
        collage_ok = create_collage(sug["items"], out_dir / fname)
        result.append({
            "name": sug["name"],
            "description": sug["description"],
            "items": [
                {"role": item["role"], "label": item["label"], "image": str(item["image"])}
                for item in sug["items"]
            ],
            "collage_url": f"/images/{user_id}/generated/{fname}" if collage_ok else None,
        })

    return {"user_id": user_id, "filters": params, "suggestions": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
