import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/query"
NUM_REQUESTS = 20
CONCURRENCY = 5

TEST_QUESTIONS = [
    "What are the symptoms of cassava mosaic disease?",
    "How is cassava bacterial blight treated?",
    "What causes cassava brown streak virus?",
    "Which fungicides work against cassava leaf diseases?",
]

def send_request(i):
    question = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
    start = time.perf_counter()
    resp = requests.post(API_URL, json={"question": question, "top_k": 5})
    elapsed = time.perf_counter() - start
    return elapsed, resp.status_code

print(f"Running {NUM_REQUESTS} requests at concurrency {CONCURRENCY}...")
latencies = []
errors = 0

start_total = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = [executor.submit(send_request, i) for i in range(NUM_REQUESTS)]
    for future in futures:
        elapsed, status = future.result()
        if status != 200:
            errors += 1
        else:
            latencies.append(elapsed)
total_time = time.perf_counter() - start_total

print("\n=== Load test results ===")
print(f"Total requests     : {NUM_REQUESTS}")
print(f"Concurrency level  : {CONCURRENCY}")
print(f"Errors             : {errors}")
print(f"Total wall time    : {total_time:.2f}s")
print(f"Throughput         : {NUM_REQUESTS/total_time:.2f} req/s")
if latencies:
    print(f"Avg latency        : {statistics.mean(latencies):.2f}s")
    print(f"Median latency     : {statistics.median(latencies):.2f}s")
    print(f"P95 latency        : {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
    print(f"Min / Max          : {min(latencies):.2f}s / {max(latencies):.2f}s")

""" PS C:\Users\PC\Desktop\a\pfe\cassava-agent> & c:/Users/PC/Desktop/a/pfe/cassava-agent/.venv/Scripts/python.exe c:/Users/PC/Desktop/a/pfe/cassava-agent/scripts/load_test.py
Running 20 requests at concurrency 5...

=== Load test results ===
Total requests     : 20
Concurrency level  : 5
Errors             : 2
Total wall time    : 20.04s
Throughput         : 1.00 req/s
Avg latency        : 5.07s
Median latency     : 2.04s
P95 latency        : 12.41s
Min / Max          : 2.02s / 12.41s
(.venv) PS C:\Users\PC\Desktop\a\pfe\cassava-agent> """