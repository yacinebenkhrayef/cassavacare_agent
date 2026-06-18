from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "cassavacare_docs"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension

client = QdrantClient(host="localhost", port=6333)

# Check if it already exists to make the script idempotent
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
else:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")

# Verify
info = client.get_collection(COLLECTION_NAME)
print(f"Status: {info.status}")
print(f"Points count: {info.points_count}")