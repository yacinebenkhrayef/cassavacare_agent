from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

COLLECTION_NAME = "cassavacare_docs"
client = QdrantClient(host="localhost", port=6333)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="source",
    field_schema=PayloadSchemaType.KEYWORD,
)
print("Payload index on 'source' created.")