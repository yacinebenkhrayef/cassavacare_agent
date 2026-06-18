import requests
import json

API_URL = "http://localhost:8000/query"

TEST_CASES = [
    {"question": "What are the symptoms of cassava mosaic disease?", "top_k": 5},
    {"question": "How is cassava bacterial blight treated?", "top_k": 5},
    {"question": "What causes cassava brown streak virus?", "top_k": 3},
    {"question": "Which fungicides are recommended for cassava leaf diseases?", "top_k": 5},
    {"question": "How do I prevent cassava green mottle?", "top_k": 4},
]

print("=" * 70)
for tc in TEST_CASES:
    resp = requests.post(API_URL, json=tc)
    data = resp.json()
    print(f"\nQ: {tc['question']}")
    print(f"A: {data['answer']}")
    print(f"Sources: {[s['source'] for s in data['sources']]}")
    print(f"Scores:  {[s['score'] for s in data['sources']]}")
    print("-" * 70)