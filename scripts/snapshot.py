from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333, timeout=60.0)

print("Triggering background snapshot creation...")

# Trigger the background snapshot
client.create_snapshot(collection_name="cassavacare_docs", wait=False)

print("\n--- Success ---")
print("Snapshot job accepted by the server!")
print("-" * 20)

# Fetch the active snapshot list to see your files
print("\nFetching active snapshot list from server:")
try:
    existing_snapshots = client.list_snapshots(collection_name="cassavacare_docs")
    if not existing_snapshots:
        print("No finalized snapshots found yet (it might still be cooking in the background).")
    for snap in existing_snapshots:
        print(f" - {snap.name} ({snap.size / 1024 / 1024:.2f} MB) | Created: {snap.creation_time}")
except Exception as e:
    print(f"Could not retrieve snapshot list: {e}")