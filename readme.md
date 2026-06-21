# AI Wardrobe Stylist

A local FastAPI service that scans a wardrobe of clothing images, scores every item
against style / season / weather filters using the CLIP vision model, and returns up
to three outfit suggestions with a composed collage image for each.

---

## Requirements

- Python 3.10 or newer
- ~600 MB free disk space (CLIP model weights)
- No GPU required — everything runs on CPU

---

## Setup

### 1. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# CPU-only PyTorch (saves ~600 MB — skip if you already have torch installed)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Rest of the dependencies
pip install -r requirements.txt
```

### 3. Download the CLIP model

Run once — weights (~340 MB) are saved to the local `model/` folder.

```bash
python download_model.py
```

### 4. Start the server

```bash
python main.py
```

The API is now available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## Folder structure

```
project/
├── main.py               # FastAPI application
├── tryon.py              # Virtual try-on module (see below)
├── download_model.py     # One-time model downloader
├── requirements.txt
├── model/                # Cached CLIP weights (created by download_model.py)
│
├── BaseImages/           # All user data lives here
│   └── {user_id}/        # one folder per user  (email address recommended)
│       ├── Tops/
│       │   ├── T-shirt/
│       │   │   ├── img1.jpg
│       │   │   └── img2.jpg
│       │   └── Blouse/
│       ├── Bottoms/
│       │   ├── Jeans/
│       │   └── Trousers/
│       ├── Dress/
│       │   ├── Mini/
│       │   └── Maxi/
│       ├── Shoes/
│       ├── Outerwear/
│       ├── Accessories/
│       └── generated/    # ← outfit collages are saved here automatically
│
└── Images/               # Legacy default wardrobe (used as fallback if no user folder exists)
```

### Adding a wardrobe

**Option A — Upload a zip via the API** (see `/wardrobe/{user_id}` below).
The zip may contain any number of leading folders; the app walks inside and finds
the first directory that contains the expected category folders.

**Option B — Copy files manually**
Place images directly under `BaseImages/{user_id}/Category/Subcategory/`.

**Migrating the existing `Images/` folder**
```powershell
# PowerShell
Copy-Item -Recurse Images\ BaseImages\your@email.com\
```

---

## API reference

### `GET /health`
Returns `{"status": "ok"}`.

---

### `POST /wardrobe/{user_id}`
Upload a zip file containing the wardrobe.  
Extraction is skipped if `BaseImages/{user_id}/` already exists.

| Parameter | Type        | Description                   |
|-----------|-------------|-------------------------------|
| `user_id` | path string | User identifier (e.g. email)  |
| `file`    | form file   | `.zip` archive of the wardrobe |

**Example (curl)**
```bash
curl -X POST "http://localhost:8000/wardrobe/user@example.com" \
     -F "file=@my_wardrobe.zip"
```

---

### `GET /outfit/{user_id}`
Generate up to 3 outfit suggestions scored by CLIP.

| Parameter | Type   | Values                                    | Default   |
|-----------|--------|-------------------------------------------|-----------|
| `user_id` | path   | any registered user id                    | —         |
| `style`   | query  | `casual` `formal` `elegant` `sporty`      | `casual`  |
| `season`  | query  | `winter` `spring` `summer` `autumn`       | `spring`  |
| `weather` | query  | `sunny` `cloudy` `rainy` `cold`           | `sunny`   |

**Example**
```
GET /outfit/user@example.com?style=elegant&season=summer&weather=sunny
```

**Response**
```json
{
  "user_id": "user@example.com",
  "filters": { "style": "elegant", "season": "summer", "weather": "sunny" },
  "suggestions": [
    {
      "name": "Evening Elegance",
      "description": "A elegant outfit for summer, sunny weather.",
      "items": [
        { "role": "dress",     "label": "Dress/Mini",         "image": "..." },
        { "role": "shoes",     "label": "Shoes/Sandals",      "image": "..." },
        { "role": "accessory", "label": "Accessories/Jewelry","image": "..." }
      ],
      "collage_url": "/images/user@example.com/generated/outfit_elegant_summer_sunny_1.jpg"
    }
  ]
}
```

Collages are served as static files.  
Full URL: `http://localhost:8000/images/{user_id}/generated/{filename}`

---

## How outfit scoring works

1. Every image in the wardrobe is encoded into a 512-dimensional CLIP embedding
   (computed once per wardrobe path, cached in memory).
2. For a request with filters `style + season + weather`, a combined text prompt is
   built (e.g. *"elegant sophisticated evening wear. light summer outfit…"*) and
   encoded into the same 512-d space.
3. Each item is ranked by cosine similarity to the text prompt within its category.
4. The top-ranked items are combined into three outfits, varying which items are
   chosen across suggestions for variety.
5. A JPEG collage is assembled from the selected items.

