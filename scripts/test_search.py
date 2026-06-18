from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "cassavacare_docs"
TOP_K = 5

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query: str, source_filter: str = None):
    query_vector = model.encode(query).tolist()

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    # Change client.search to client.query_points
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,       # Note: the argument name is 'query', not 'query_vector'
        limit=TOP_K,
        query_filter=search_filter,
        with_payload=True,
    )
    
    # query_points returns a Response object; the actual list of hits is in .points
    return response.points

# --- Test queries ---
queries = [
    "symptoms of cassava mosaic disease",
    "treatment for cassava bacterial blight",
    "how to prevent brown streak virus",
    "fungicide application for cassava leaf disease",
]

for q in queries:
    print(f"\nQuery: {q}")
    print("-" * 60)
    results = search(q)
    for r in results:
        print(f"  Score: {r.score:.4f} | Source: {r.payload['source']}")
        print(f"  Text : {r.payload['text'][:120]}...")
        print()