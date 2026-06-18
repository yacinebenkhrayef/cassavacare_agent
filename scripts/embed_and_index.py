import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

COLLECTION_NAME = "cassavacare_docs"
CHUNKS_PATH = "documents/chunks/all_chunks.json"
BATCH_SIZE = 64  # number of chunks to embed + upsert at once

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load chunks
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")

# Process in batches
points = []
for i, chunk in enumerate(tqdm(chunks, desc="Embedding")):
    vector = model.encode(chunk["text"]).tolist()

    point = PointStruct(
        id=chunk["id"],
        vector=vector,
        payload={
            "text": chunk["text"],
            "source": chunk["source"],
            "filename": chunk["filename"],
        }
    )
    points.append(point)

    # Upsert when batch is full or at the end
    if len(points) == BATCH_SIZE or i == len(chunks) - 1:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        points = []

print("\nIndexing complete.")
info = client.get_collection(COLLECTION_NAME)
print(f"Total vectors in collection: {info.points_count}")