from langchain_text_splitters import RecursiveCharacterTextSplitter
import json, os

CLEANED_DIR = "documents/cleaned"
CHUNKS_DIR = "documents/chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

all_chunks = []
chunk_id = 0

# Process .txt files (FAO, IITA, Wikipedia)
for fname in os.listdir(CLEANED_DIR):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(CLEANED_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        text = f.read()

    # Infer source from filename
    if fname.startswith("fao"):
        source = "FAO"
    elif fname.startswith("iita"):
        source = "IITA"
    elif fname.startswith("wiki"):
        source = "Wikipedia"
    else:
        source = "unknown"

    chunks = splitter.split_text(text)
    for chunk in chunks:
        if len(chunk) < 100:
            continue
        all_chunks.append({
            "id": chunk_id,
            "source": source,
            "filename": fname,
            "text": chunk
        })
        chunk_id += 1

# Process PubMed JSON (each abstract = 1 chunk, already short)
with open(os.path.join(CLEANED_DIR, "pubmed_cleaned.json"), encoding="utf-8") as f:
    abstracts = json.load(f)

for item in abstracts:
    if len(item["text"]) < 100:
        continue
    all_chunks.append({
        "id": chunk_id,
        "source": "PubMed",
        "filename": f"pmid_{item['pmid']}",
        "text": item["text"]
    })
    chunk_id += 1

# Save
output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"\nTotal chunks created: {len(all_chunks)}")
print(f"Saved to: {output_path}")