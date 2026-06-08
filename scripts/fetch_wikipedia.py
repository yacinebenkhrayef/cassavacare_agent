import requests
from bs4 import BeautifulSoup
import json, os

PAGES = [
    "Cassava_mosaic_virus",
    "Cassava_brown_streak_virus",
    "Cassava_bacterial_blight",
    "Cassava_green_mottle_virus",
    "Mancozeb",
    "Fungicide",
    "Plant_disease_resistance",
]

OUTPUT_DIR = "documents/raw/wikipedia"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for page in PAGES:
    url = f"https://en.wikipedia.org/wiki/{page}"
    resp = requests.get(url, headers={"User-Agent": "CassavaCare-RAG/1.0"})
    soup = BeautifulSoup(resp.text, "lxml")

    # Extract only the main content paragraphs
    content_div = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content_div.find_all("p") if content_div else []
    text = "\n\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 60)

    out_path = f"{OUTPUT_DIR}/{page.lower()}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved: {out_path} ({len(text)} chars)")