from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from api.models import SourceChunk
from typing import List, Optional

COLLECTION_NAME = "cassavacare_docs"

_client = None
_model = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host="localhost", port=6333)
    return _client

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model ready.")
    return _model

def retrieve(question: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[SourceChunk]:
    model = get_model()
    client = get_client()

    query_vector = model.encode(question).tolist()

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    # --- CHANGED HERE: Using modern query_points API ---
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    ).points  # Make sure to append .points here to extract the array of records

    return [
        SourceChunk(
            text=r.payload["text"],
            source=r.payload["source"],
            filename=r.payload["filename"],
            score=round(r.score, 4),
        )
        for r in results
    ]