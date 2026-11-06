import time
import statistics
import requests

URL = "http://127.0.0.1:8000/chat"
USER_ID = 1
REQUESTS_TO_RUN = 10

payload = {
    "user_id": USER_ID,
    "message": "Where is my order ORD1001?"
}

latencies = []
successes = 0
errors = 0

print("Starting API benchmark...")
print(f"Endpoint: {URL}")
print(f"Requests: {REQUESTS_TO_RUN}")
print()

for i in range(REQUESTS_TO_RUN):
    start = time.perf_counter()
    try:
        response = requests.post(URL, json=payload, timeout=90)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if response.status_code == 200:
            successes += 1
            print(f"Request {i + 1}: HTTP {response.status_code}, {elapsed:.0f} ms")
        else:
            errors += 1
            print(f"Request {i + 1}: HTTP {response.status_code}, {elapsed:.0f} ms")
    except requests.RequestException as exc:
        elapsed = (time.perf_counter() - start) * 1000
        errors += 1
        print(f"Request {i + 1}: ERROR after {elapsed:.0f} ms ({type(exc).__name__})")

print("\n===== BENCHMARK RESULTS =====")
print(f"Total requests: {REQUESTS_TO_RUN}")
print(f"Successful HTTP 200 responses: {successes}")
print(f"Errors/non-200 responses: {errors}")
print(f"HTTP success rate: {successes / REQUESTS_TO_RUN * 100:.1f}%")
print(f"API error rate: {errors / REQUESTS_TO_RUN * 100:.1f}%")

if latencies:
    print(f"Mean latency: {statistics.mean(latencies):.1f} ms")
    print(f"Median latency: {statistics.median(latencies):.1f} ms")
    print(f"Minimum latency: {min(latencies):.1f} ms")
    print(f"Maximum latency: {max(latencies):.1f} ms")
    ordered = sorted(latencies)
    p95_index = max(0, __import__("math").ceil(0.95 * len(ordered)) - 1)
    print(f"Approximate p95 latency: {ordered[p95_index]:.1f} ms")


