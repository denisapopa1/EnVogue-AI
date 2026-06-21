import os
import shutil
import sqlite3
import urllib.parse
import uuid
from pathlib import Path

DB_PATH = Path(r"C:\Desktop\chestii pt licenta\WardrobeApi\WardrobeApi\Wardrobe.db")
UPLOADS = Path(r"C:\Desktop\chestii pt licenta\WardrobeApi\WardrobeApi\wwwroot\uploads\images")
MODEL_IMAGES = Path(r"C:\Desktop\chestii pt licenta\Model preantrenat\BaseImages")


def main() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    rows = connection.execute(
        "select Id, ImagePath from ClothingItems where ImagePath like 'http://localhost:8000/images/%'"
    ).fetchall()

    updated = 0
    missing = 0
    for item_id, url in rows:
        parsed = urllib.parse.urlparse(url)
        relative = urllib.parse.unquote(parsed.path.removeprefix("/images/")).replace("/", os.sep)
        source = MODEL_IMAGES / relative

        if not source.exists():
            missing += 1
            print(f"missing item={item_id} source={source}")
            continue

        extension = source.suffix or ".jpg"
        file_name = f"{uuid.uuid4()}{extension}"
        shutil.copy2(source, UPLOADS / file_name)
        connection.execute("update ClothingItems set ImagePath = ? where Id = ?", (file_name, item_id))
        updated += 1

    connection.commit()
    print(f"updated={updated} missing={missing}")


if __name__ == "__main__":
    main()
