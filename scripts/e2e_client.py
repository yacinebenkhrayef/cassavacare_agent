"""
Phase 4, Part 4 — end-to-end client.

Drives the live CassavaCare-Agent API (running standalone or via
docker-compose) through the full pipeline for every image in the manifest
produced by scripts/sample_test_images.py, and records:
  - wall-clock time from submit (202 Accepted) to status == "completed"
  - the full DiagnosisResult for each job, for the independent oracle check
    in scripts/validate_decision_accuracy.py

This exercises the REAL running system exactly as a user/dashboard would
experience it. Contrast with tests/test_agent_reliability.py (§8), which
invokes agent_graph directly with controlled inputs to guarantee every
§6.3 branch is covered regardless of the day's actual weather.
"""
import time
from pathlib import Path

import pandas as pd
import requests

API_BASE_URL = "http://localhost:8000"            # match your docker-compose port mapping
MANIFEST_PATH = "data/e2e_test_set/manifest.csv"
OUTPUT_DIR = Path("reports/e2e_run")
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 60.0

# Rotate through a few cities so weather branches vary organically across the
# 30 images instead of every job hitting the same 15-minute OpenWeather cache
# entry (see Part 2, WEATHER_CACHE_TTL_SECONDS) for one single city.
TEST_CITIES = ["Tunis,TN", "Sfax,TN", "Sousse,TN", "Bizerte,TN", "Kairouan,TN"]

# Gemini's free tier is rate-limited (see Part 3 §4 note) — pace requests so
# a 30-image run doesn't spend most of its time inside GeminiClient's
# retry/backoff loop.
SECONDS_BETWEEN_SUBMISSIONS = 4.0


def submit_job(image_path: str, location: str) -> str:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{API_BASE_URL}/diagnose",
            files={"image": (Path(image_path).name, f, "image/jpeg")},
            data={"location": location},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll_until_done(job_id: str) -> tuple[dict, float]:
    start = time.monotonic()
    while True:
        resp = requests.get(f"{API_BASE_URL}/diagnose/{job_id}", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body, time.monotonic() - start
        if time.monotonic() - start > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(f"Job {job_id} did not finish within {POLL_TIMEOUT_SECONDS}s")
        time.sleep(POLL_INTERVAL_SECONDS)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, row in manifest.iterrows():
        location = TEST_CITIES[i % len(TEST_CITIES)]
        print(f"[{i + 1}/{len(manifest)}] {row['image_path']} -> {location}")

        job_id = submit_job(row["image_path"], location)
        body, elapsed_s = poll_until_done(job_id)
        result = body.get("result") or {}

        rows.append({
            "image_path": row["image_path"],
            "true_label": row.get("true_label"),
            "location": location,
            "job_id": job_id,
            "status": body["status"],
            "elapsed_seconds": round(elapsed_s, 2),
            "pred_disease_short": result.get("pred_disease_short"),
            "confidence": result.get("confidence"),
            "needs_new_image": result.get("needs_new_image"),
            "rain_probability": (result.get("weather") or {}).get("rain_probability"),
            "wind_speed_kmh": (result.get("weather") or {}).get("wind_speed_kmh"),
            "weather_error": result.get("weather_error"),
            "decision": result.get("decision"),
            "decision_reason": result.get("decision_reason"),
            "num_rag_sources": len(result.get("rag_sources") or []),
            "final_report_is_fallback": (result.get("final_report") or "").startswith("[Fallback report"),
        })

        time.sleep(SECONDS_BETWEEN_SUBMISSIONS)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "e2e_results.csv", index=False)

    # --- §6.4 KPI: response time (submit -> completed), reinterpreted for
    # the async job design — see §0 flag above. ---
    times = df["elapsed_seconds"]
    under_10s_pct = (times < 10).mean() * 100
    print("\n--- §6.4 Response time (submit -> completed) ---")
    print(f"mean={times.mean():.2f}s  median={times.median():.2f}s  "
          f"p95={times.quantile(0.95):.2f}s  max={times.max():.2f}s")
    print(f"% under 10s: {under_10s_pct:.1f}%  (target from §6.4/§8, reinterpreted for async polling)")

    failed = (df["status"] == "failed").sum()
    if failed:
        print(f"\n⚠️  {failed} job(s) failed — inspect reports/e2e_run/e2e_results.csv before trusting the score.")

    print(f"\nSaved {len(df)} rows to {OUTPUT_DIR / 'e2e_results.csv'}")
    print("Run scripts/validate_decision_accuracy.py next for the §6.3 score.")


if __name__ == "__main__":
    main()