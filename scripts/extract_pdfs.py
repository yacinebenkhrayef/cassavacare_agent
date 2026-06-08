import fitz  # PyMuPDF
import os, re

RAW_DIRS = ["documents/raw/fao", "documents/raw/iita"]
OUTPUT_DIR = "documents/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)       # collapse excessive blank lines
    text = re.sub(r'[ \t]+', ' ', text)            # collapse spaces/tabs
    text = re.sub(r'\f', '\n\n', text)             # form feeds → paragraph break
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # fix hyphenated line breaks
    return text.strip()

for raw_dir in RAW_DIRS:
    source_name = os.path.basename(raw_dir)
    for fname in os.listdir(raw_dir):
        if not fname.endswith(".pdf"):
            continue
        pdf_path = os.path.join(raw_dir, fname)
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()

        cleaned = clean_text(full_text)
        out_name = fname.replace(".pdf", ".txt")
        out_path = os.path.join(OUTPUT_DIR, f"{source_name}_{out_name}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"Extracted: {out_path} ({len(cleaned)} chars)")