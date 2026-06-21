import sqlite3

DB_PATH = r"C:\Desktop\chestii pt licenta\WardrobeApi\WardrobeApi\Wardrobe.db"

connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
localhost_count = connection.execute(
    "select count(*) from ClothingItems where ImagePath like ?",
    ("%localhost%",),
).fetchone()[0]
latest = connection.execute(
    "select Id, ImagePath from ClothingItems order by Id desc limit 5"
).fetchall()

print(f"localhost paths={localhost_count}")
print(latest)
