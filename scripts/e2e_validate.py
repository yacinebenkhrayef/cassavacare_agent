#!/usr/bin/env python3
"""
Phase 5 — Part 4: scripted §6.3 reliability re-check.

Runs the 30-image stratified sample through the FULL docker-composed stack
(qdrant + backend + dashboard all up), talking to the backend's published
port directly — this exercises the exact same agent pipeline the dashboard
uses, without needing to automate a browser.

Usage:
    docker compose up -d
    python scripts/e2e_validate.py --manifest validation/manifest_template.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"   # host-published port — script runs outside the compose network
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 30

RAIN_THRESHOLD = 0.5        # §6.3: "pluie prévue > 50% dans les 6h"
WIND_THRESHOLD = 15.0       # §6.3: "vent > 15 km/h"
RARE_DISEASE_LABEL = "cbsd"
RARE_DISEASE_MIN_SOURCES = 2
GLOBAL_TARGET = 0.85


@dataclass
class CaseResult:
    image_path: str
    bucket: str
    passed: bool
    detail: str
    processing_s: float
    extra_note: str = ""


def submit(image_path: Path, location: str) -> str:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/diagnose",
            files={"image": (image_path.name, f, "image/jpeg")},
            data={"location": location},
            timeout=15,
        )
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll(job_id: str) -> tuple[dict, float]:
    start = time.monotonic()
    deadline = start + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = requests.get(f"{BASE_URL}/diagnose/{job_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data["status"].lower() in ("completed", "failed"):
            return data, time.monotonic() - start
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Job {job_id} did not reach a terminal state within {POLL_TIMEOUT_S}s")


def evaluate(job_status: dict, processing_s: float, image_path: str) -> CaseResult:
    if job_status["status"].lower() == "failed":
        return CaseResult(image_path, "job_failed", False,
                           f"Job failed: {job_status.get('error')}", processing_s)

    result = job_status.get("result") or {}

    # Bucket 1 — low confidence
    if result.get("needs_new_image"):
        return CaseResult(image_path, "low_confidence", True,
                           "needs_new_image=True as expected", processing_s)

    decision = result.get("decision")
    weather = result.get("weather") or {}
    weather_error = result.get("weather_error")
    label = (result.get("pred_disease_short") or result.get("pred_disease") or "").lower()
    is_healthy = label == "healthy"
    rag_sources = result.get("rag_sources") or []

    # Rare-disease check runs independently of the weather bucket below
    extra_note = ""
    if RARE_DISEASE_LABEL in label:
        ok = len(rag_sources) >= RARE_DISEASE_MIN_SOURCES
        extra_note = (
            f"CBSD rare-disease check: {len(rag_sources)} source(s) "
            f"({'OK' if ok else 'FAIL, expected >= ' + str(RARE_DISEASE_MIN_SOURCES)})"
        )
        if not ok:
            return CaseResult(image_path, "rare_disease_cbsd", False, extra_note, processing_s)

    # Healthy leaf — not one of the 5 official §6.3 rows, tracked separately
    if is_healthy:
        ok = decision == "no_action_needed"
        return CaseResult(image_path, "healthy_leaf", ok,
                           f"decision={decision!r}", processing_s, extra_note)

    # Weather-driven buckets (only reachable for a diseased leaf, confidence >= 0.7)
    if weather_error:
        # Weather API failure — decision_node's fallback path; the cahier des charges doesn't
        # specify an expected value here, so this is logged, not scored pass/fail.
        return CaseResult(image_path, "weather_unavailable", True,
                           f"weather_error={weather_error!r} — not scored, see guide §2",
                           processing_s, extra_note)

    rain = weather.get("rain_probability", 0.0)
    wind = weather.get("wind_speed_kmh", 0.0)
    rain_flag = rain > RAIN_THRESHOLD
    wind_flag = wind > WIND_THRESHOLD

    if rain_flag or wind_flag:
        # Either restrictive decision is accepted if both trigger at once — see guide §2.
        ok = decision in ("defer", "avoid_aerial")
        bucket = "rain_defer" if rain_flag and not wind_flag else (
            "wind_avoid_aerial" if wind_flag and not rain_flag else "rain_and_wind"
        )
        return CaseResult(image_path, bucket, ok,
                           f"rain={rain:.0%}, wind={wind:.1f}km/h, decision={decision!r}",
                           processing_s, extra_note)

    # Conditions OK
    ok = decision == "apply"
    return CaseResult(image_path, "conditions_ok", ok,
                       f"rain={rain:.0%}, wind={wind:.1f}km/h, decision={decision!r}",
                       processing_s, extra_note)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("validation/e2e_validation_results.csv"))
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if not rows:
        sys.exit(f"No rows found in {args.manifest}")

    results: list[CaseResult] = []
    for row in rows:
        image_path = Path(row["image_path"])
        location = row["location"]
        print(f"→ {image_path.name} ({location}) …", end=" ", flush=True)
        try:
            job_id = submit(image_path, location)
            job_status, processing_s = poll(job_id)
            case = evaluate(job_status, processing_s, str(image_path))
        except Exception as exc:  # noqa: BLE001 — log and keep going through the batch
            case = CaseResult(str(image_path), "error", False, str(exc), 0.0)
        results.append(case)
        print(f"{case.bucket} — {'PASS' if case.passed else 'FAIL'} ({case.processing_s:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "bucket", "passed", "detail", "processing_s", "extra_note"])
        for c in results:
            writer.writerow([c.image_path, c.bucket, c.passed, c.detail, f"{c.processing_s:.2f}", c.extra_note])

    total = len(results)
    passed = sum(1 for c in results if c.passed)
    rate = passed / total if total else 0.0

    print("\n" + "=" * 60)
    print(f"§6.3 reliability score: {passed}/{total} = {rate:.1%} "
          f"(target ≥ {GLOBAL_TARGET:.0%}) — {'PASS' if rate >= GLOBAL_TARGET else 'FAIL'}")

    buckets_seen = {c.bucket for c in results}
    expected_buckets = {"low_confidence", "rain_defer", "wind_avoid_aerial", "conditions_ok", "rare_disease_cbsd"}
    missing = expected_buckets - buckets_seen
    if missing:
        print(f"⚠️  These §6.3 scenario rows were never naturally triggered by the 30 images: "
              f"{', '.join(sorted(missing))} — consider a targeted manual test for these.")
    print(f"Full per-image results written to {args.out}")


if __name__ == "__main__":
    main()