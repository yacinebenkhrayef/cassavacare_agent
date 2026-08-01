"""
CassavaCare-Agent — RAG Evaluation Harness
==========================================
Computes Precision@k, Recall@k, MRR, average similarity score,
and per-stage latency across 20 test queries.

Usage
-----
  # With the FastAPI server running on localhost:8000:
  python evaluate_rag.py

  # With direct Qdrant access (no server needed):
  python evaluate_rag.py --mode direct

Output
------
  - Console report (copy directly into PFE report)
  - evaluation_results.json  (full per-query breakdown)
  - evaluation_summary.csv   (table-ready for LaTeX)
"""

import json
import time
import argparse
import statistics
import csv
from pathlib import Path

# ── Relevance judgment ───────────────────────────────────────────────────────
# A retrieved chunk is counted as relevant if its text contains at least
# MIN_KEYWORD_MATCHES of the ground-truth keywords for that query.
# This is a keyword-overlap heuristic — no manual annotation needed,
# but you can override individual judgments in MANUAL_OVERRIDES below.
MIN_KEYWORD_MATCHES = 1   # lower = more permissive; raise to 2 for stricter eval

# Optional: manually override relevance for specific (query_id, chunk_index) pairs.
# Format: { "CMD_01": {0: True, 2: False} }  means "for CMD_01, chunk at rank 0
# is relevant regardless of keywords, chunk at rank 2 is not."
MANUAL_OVERRIDES = {}

K_VALUES = [1, 3, 5]   # The CDC targets P@3 and R@5 specifically


# ── Relevance function ───────────────────────────────────────────────────────
def is_relevant(chunk_text: str, relevant_keywords: list[str], query_id: str, rank: int) -> bool:
    if query_id in MANUAL_OVERRIDES and rank in MANUAL_OVERRIDES[query_id]:
        return MANUAL_OVERRIDES[query_id][rank]
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in relevant_keywords if kw.lower() in text_lower)
    return matches >= MIN_KEYWORD_MATCHES


# ── Metrics ──────────────────────────────────────────────────────────────────
def precision_at_k(relevance_flags: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = relevance_flags[:k]
    return sum(top_k) / k

def recall_at_k(relevance_flags: list[bool], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    top_k = relevance_flags[:k]
    return sum(top_k) / total_relevant

def reciprocal_rank(relevance_flags: list[bool]) -> float:
    for i, rel in enumerate(relevance_flags):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


# ── API mode (server must be running) ────────────────────────────────────────
def query_via_api(question: str, top_k: int, api_url: str) -> dict:
    import requests
    t0 = time.perf_counter()
    resp = requests.post(
        f"{api_url}/query",
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()

    chunks = [
        {
            "text": s["text"],
            "source": s["source"],
            "score": s["score"],
        }
        for s in data["sources"]
    ]
    timing = data.get("timing_ms", {})
    timing["wall_ms"] = round(wall_ms, 2)
    return {"chunks": chunks, "answer": data.get("answer", ""), "timing": timing}


# ── Direct mode (no server, calls Qdrant + model directly) ───────────────────
def query_direct(question: str, top_k: int) -> dict:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    t_embed_start = time.perf_counter()
    vector = model.encode(question).tolist()
    embed_ms = (time.perf_counter() - t_embed_start) * 1000

    t_search_start = time.perf_counter()
    results = client.search(
        collection_name="cassavacare_docs",
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    search_ms = (time.perf_counter() - t_search_start) * 1000

    chunks = [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "score": round(r.score, 4),
        }
        for r in results
    ]
    return {
        "chunks": chunks,
        "answer": "",
        "timing": {
            "embed_ms": round(embed_ms, 2),
            "retrieval_ms": round(search_ms, 2),
            "wall_ms": round(embed_ms + search_ms, 2),
        },
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(test_set: list[dict], mode: str, api_url: str, top_k_max: int):
    results = []

    print(f"\nRunning evaluation — mode={mode}, queries={len(test_set)}, top_k_max={top_k_max}")
    print("=" * 72)

    for item in test_set:
        qid = item["id"]
        question = item["question"]
        keywords = item["relevant_keywords"]
        disease = item["disease_class"]

        print(f"  [{qid}] {question[:65]}...")

        try:
            if mode == "api":
                result = query_via_api(question, top_k_max, api_url)
            else:
                result = query_direct(question, top_k_max)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"id": qid, "error": str(e)})
            continue

        chunks = result["chunks"]
        timing = result["timing"]

        # Relevance flags for each retrieved chunk
        flags = [
            is_relevant(c["text"], keywords, qid, rank)
            for rank, c in enumerate(chunks)
        ]
        scores = [c["score"] for c in chunks]
        sources = [c["source"] for c in chunks]

        # Estimate total relevant in corpus (upper bound = all retrieved that are relevant,
        # minimum 1 to avoid division-by-zero in recall)
        total_relevant = max(sum(flags), 1)

        row = {
            "id": qid,
            "disease_class": disease,
            "question": question,
            "flags": flags,
            "scores": scores,
            "sources": sources,
            "timing": timing,
            "avg_score": round(statistics.mean(scores), 4) if scores else 0,
            "top1_score": scores[0] if scores else 0,
            "rr": round(reciprocal_rank(flags), 4),
        }
        for k in K_VALUES:
            row[f"P@{k}"] = round(precision_at_k(flags, k), 4)
            row[f"R@{k}"] = round(recall_at_k(flags, k, total_relevant), 4)

        results.append(row)

        # Per-query console summary
        p3 = row.get("P@3", 0)
        r5 = row.get("R@5", 0)
        print(f"    P@3={p3:.2f}  R@5={r5:.2f}  MRR={row['rr']:.2f}  "
              f"avg_score={row['avg_score']:.4f}  "
              f"wall={timing.get('wall_ms', '?')}ms")

    return results


# ── Aggregate reporting ───────────────────────────────────────────────────────
def aggregate(results: list[dict]) -> None:
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print("\n" + "=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)

    # Overall metrics
    for k in K_VALUES:
        pk_vals = [r[f"P@{k}"] for r in valid]
        rk_vals = [r[f"R@{k}"] for r in valid]
        print(f"  mean P@{k}: {statistics.mean(pk_vals):.4f}   "
              f"mean R@{k}: {statistics.mean(rk_vals):.4f}")

    mrr_vals = [r["rr"] for r in valid]
    print(f"  MRR (mean reciprocal rank): {statistics.mean(mrr_vals):.4f}")
    avg_scores = [r["avg_score"] for r in valid]
    print(f"  Mean avg similarity score:  {statistics.mean(avg_scores):.4f}")
    top1_scores = [r["top1_score"] for r in valid]
    print(f"  Mean top-1 similarity score:{statistics.mean(top1_scores):.4f}")

    # Latency
    wall_times = [r["timing"].get("wall_ms", 0) for r in valid if "timing" in r]
    if wall_times:
        print(f"\n  Latency (wall, ms):")
        print(f"    mean   : {statistics.mean(wall_times):.1f}")
        print(f"    median : {statistics.median(wall_times):.1f}")
        sorted_w = sorted(wall_times)
        p95_idx = int(len(sorted_w) * 0.95)
        print(f"    P95    : {sorted_w[p95_idx]:.1f}")
        print(f"    min    : {min(wall_times):.1f}")
        print(f"    max    : {max(wall_times):.1f}")

    # Per-disease breakdown
    print("\n  Per-disease breakdown (P@3 / R@5 / MRR):")
    diseases = sorted(set(r["disease_class"] for r in valid))
    for disease in diseases:
        subset = [r for r in valid if r["disease_class"] == disease]
        p3 = statistics.mean(r["P@3"] for r in subset)
        r5 = statistics.mean(r["R@5"] for r in subset)
        mrr = statistics.mean(r["rr"] for r in subset)
        print(f"    {disease:<35} P@3={p3:.4f}  R@5={r5:.4f}  MRR={mrr:.4f}")

    if errors:
        print(f"\n  ERRORS ({len(errors)} queries failed):")
        for e in errors:
            print(f"    [{e['id']}]: {e['error']}")

    print("=" * 72)


# ── Export results ────────────────────────────────────────────────────────────
def export_json(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved → {path}")


def export_csv(results: list[dict], path: str) -> None:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return
    fields = (
        ["id", "disease_class"]
        + [f"P@{k}" for k in K_VALUES]
        + [f"R@{k}" for k in K_VALUES]
        + ["rr", "avg_score", "top1_score"]
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)
    print(f"CSV summary saved    → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CassavaCare RAG evaluation harness")
    parser.add_argument(
        "--mode",
        choices=["api", "direct"],
        default="api",
        help="'api' uses the FastAPI server; 'direct' calls Qdrant + model directly",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI server (only used in api mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum chunks to retrieve per query (must be >= max K_VALUE)",
    )
    parser.add_argument(
        "--test-set",
        default="test_set.json",
        help="Path to the test_set.json file",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write output files",
    )
    args = parser.parse_args()

    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        print(f"ERROR: test set not found at {test_set_path}")
        exit(1)

    with open(test_set_path, encoding="utf-8") as f:
        test_set = json.load(f)

    results = run_evaluation(
        test_set=test_set,
        mode=args.mode,
        api_url=args.api_url,
        top_k_max=args.top_k,
    )

    aggregate(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_json(results, str(out_dir / "evaluation_results.json"))
    export_csv(results, str(out_dir / "evaluation_summary.csv"))