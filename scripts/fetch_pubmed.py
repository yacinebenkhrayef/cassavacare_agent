from Bio import Entrez
import json, time, os
from tqdm import tqdm

Entrez.email = "your_email@example.com"  # required by NCBI

QUERIES = [
    "cassava mosaic disease treatment",
    "cassava bacterial blight management",
    "cassava brown streak disease control",
    "cassava green mottle fungicide",
    "Manihot esculenta disease resistance",
]

OUTPUT_DIR = "documents/raw/pubmed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_abstracts = []

for query in QUERIES:
    print(f"\nSearching: {query}")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
    record = Entrez.read(handle)
    ids = record["IdList"]

    for pmid in tqdm(ids):
        try:
            fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
            abstract_text = fetch.read()
            all_abstracts.append({
                "pmid": pmid,
                "query": query,
                "text": abstract_text.strip()
            })
            time.sleep(0.4)  # respect NCBI rate limit (max 3 req/sec)
        except Exception as e:
            print(f"  Error on PMID {pmid}: {e}")

with open(f"{OUTPUT_DIR}/pubmed_abstracts.json", "w", encoding="utf-8") as f:
    json.dump(all_abstracts, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(all_abstracts)} abstracts saved.")