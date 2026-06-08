import json, os, re

def clean(text):
    text = re.sub(r'\[\d+\]', '', text)           # remove citation brackets [1], [23]
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

OUTPUT_DIR = "documents/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clean PubMed
with open("documents/raw/pubmed/pubmed_abstracts.json", encoding="utf-8") as f:
    abstracts = json.load(f)

cleaned_abstracts = []
for item in abstracts:
    cleaned_abstracts.append({
        "pmid": item["pmid"],
        "source": "pubmed",
        "text": clean(item["text"])
    })

with open(f"{OUTPUT_DIR}/pubmed_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_abstracts, f, ensure_ascii=False, indent=2)
print(f"Cleaned {len(cleaned_abstracts)} PubMed abstracts.")

# Clean Wikipedia .txt files
for fname in os.listdir("documents/raw/wikipedia"):
    if not fname.endswith(".txt"):
        continue
    with open(f"documents/raw/wikipedia/{fname}", encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean(raw)
    with open(f"{OUTPUT_DIR}/wiki_{fname}", "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"Cleaned: wiki_{fname}")