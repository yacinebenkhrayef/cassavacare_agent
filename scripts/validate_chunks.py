import json

with open("documents/chunks/all_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

lengths = [len(c["text"]) for c in chunks]
sources = {}
for c in chunks:
    sources[c["source"]] = sources.get(c["source"], 0) + 1

print("=== Corpus Validation Report ===")
print(f"Total chunks      : {len(chunks)}")
print(f"Avg chunk length  : {sum(lengths)/len(lengths):.0f} chars")
print(f"Min chunk length  : {min(lengths)} chars")
print(f"Max chunk length  : {max(lengths)} chars")
print(f"\nChunks by source:")
for src, count in sorted(sources.items()):
    print(f"  {src:<15} {count}")

# Flag potential issues
short_chunks = [c for c in chunks if len(c["text"]) < 100]
long_chunks   = [c for c in chunks if len(c["text"]) > 600]
print(f"\nWarnings:")
print(f"  Too short (<100 chars) : {len(short_chunks)}")
print(f"  Too long  (>600 chars) : {len(long_chunks)}")

if len(chunks) < 50:
    print("\n  [!] Corpus seems small — consider adding more documents.")
if len(sources) < 3:
    print("\n  [!] Only {len(sources)} source(s) — aim for at least 3 for diversity.")
else:
    print("\n  Corpus looks healthy. Ready for Part 2.")