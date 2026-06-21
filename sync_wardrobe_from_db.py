import shutil
import sqlite3
from pathlib import Path

USER_ID = "9cb93ac1-677e-42d0-88a1-4ac33db5f6e8"
USER_EMAIL = "denisa@gmail.com"

BASE_DIR = Path(__file__).parent
TARGET_BASE = (BASE_DIR / "BaseImages").resolve()
TARGET = (TARGET_BASE / USER_EMAIL).resolve()
UPLOADS = Path(r"C:\Desktop\chestii pt licenta\WardrobeApi\WardrobeApi\wwwroot\uploads\images")
DB_PATH = Path(r"C:\Desktop\chestii pt licenta\WardrobeApi\WardrobeApi\Wardrobe.db")
INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def safe_part(value: str | None, fallback: str) -> str:
    cleaned = "".join(ch for ch in (value or fallback) if ch not in INVALID_FILENAME_CHARS).strip()
    return cleaned or fallback


def main() -> None:
    if not str(TARGET).lower().startswith(str(TARGET_BASE).lower()):
        raise RuntimeError(f"Refusing to write outside {TARGET_BASE}: {TARGET}")

    shutil.rmtree(TARGET, ignore_errors=True)

    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = connection.execute(
        """
        select
            ci.Id,
            ci.ImagePath,
            coalesce(c.Name, 'Other') as Category,
            coalesce(s.Name, 'Items') as Subcategory
        from ClothingItems ci
        left join Categories c on ci.CategoryId = c.Id
        left join Subcategories s on ci.SubcategoryId = s.Id
        where ci.UserId = ?
        """,
        (USER_ID,),
    ).fetchall()

    copied = 0
    missing = 0
    for item_id, image_path, category, subcategory in rows:
        if not image_path:
            missing += 1
            continue

        source = UPLOADS / Path(image_path).name
        if not source.exists():
            missing += 1
            continue

        destination_dir = TARGET / safe_part(category, "Other") / safe_part(subcategory, "Items")
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / f"{item_id}_{source.name}")
        copied += 1

    print(f"copied={copied} missing={missing} target={TARGET}")


if __name__ == "__main__":
    main()
