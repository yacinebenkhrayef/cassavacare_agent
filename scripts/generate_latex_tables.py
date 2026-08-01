"""
CassavaCare-Agent — LaTeX Table Generator
==========================================
Reads evaluation_results.json produced by evaluate_rag.py
and generates two LaTeX table fragments ready to paste into your PFE report:

  1. Overall metrics table (P@1, P@3, P@5, R@1, R@3, R@5, MRR, avg score)
  2. Per-disease breakdown table (P@3 / R@5 / MRR for each disease class)

Usage
-----
  python generate_latex_tables.py
  python generate_latex_tables.py --results path/to/evaluation_results.json
"""

import json
import statistics
import argparse
from pathlib import Path

K_VALUES = [1, 3, 5]


def load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [r for r in json.load(f) if "error" not in r]


def overall_table(results: list[dict]) -> str:
    rows = []
    for k in K_VALUES:
        pk = statistics.mean(r[f"P@{k}"] for r in results)
        rk = statistics.mean(r[f"R@{k}"] for r in results)
        rows.append((k, pk, rk))

    mrr = statistics.mean(r["rr"] for r in results)
    avg_score = statistics.mean(r["avg_score"] for r in results)
    top1_score = statistics.mean(r["top1_score"] for r in results)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{RAG Retrieval Evaluation — Overall Metrics (20 test queries)}",
        r"\label{tab:rag_overall}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{k=1} & \textbf{k=3} & \textbf{k=5} \\",
        r"\hline",
    ]

    pk_cells = " & ".join(f"{r[1]:.4f}" for r in rows)
    rk_cells = " & ".join(f"{r[2]:.4f}" for r in rows)
    lines.append(f"Precision@k & {pk_cells} \\\\")
    lines.append(f"Recall@k    & {rk_cells} \\\\")
    lines += [
        r"\hline",
        f"MRR (mean reciprocal rank) & \\multicolumn{{3}}{{c}}{{{mrr:.4f}}} \\\\",
        f"Mean avg similarity score  & \\multicolumn{{3}}{{c}}{{{avg_score:.4f}}} \\\\",
        f"Mean top-1 similarity score& \\multicolumn{{3}}{{c}}{{{top1_score:.4f}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def disease_table(results: list[dict]) -> str:
    diseases = sorted(set(r["disease_class"] for r in results))

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{RAG Retrieval Evaluation — Per-Disease Breakdown}",
        r"\label{tab:rag_disease}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Disease class} & \textbf{P@3} & \textbf{R@5} & \textbf{MRR} \\",
        r"\hline",
    ]

    for disease in diseases:
        subset = [r for r in results if r["disease_class"] == disease]
        p3  = statistics.mean(r["P@3"] for r in subset)
        r5  = statistics.mean(r["R@5"] for r in subset)
        mrr = statistics.mean(r["rr"]  for r in subset)
        # Escape ampersand-like characters in disease names for LaTeX
        disease_tex = disease.replace("&", r"\&")
        lines.append(f"{disease_tex} & {p3:.4f} & {r5:.4f} & {mrr:.4f} \\\\")

    # Macro-average row
    p3_all  = statistics.mean(r["P@3"] for r in results)
    r5_all  = statistics.mean(r["R@5"] for r in results)
    mrr_all = statistics.mean(r["rr"]  for r in results)
    lines += [
        r"\hline",
        f"\\textbf{{Macro average}} & \\textbf{{{p3_all:.4f}}} & \\textbf{{{r5_all:.4f}}} & \\textbf{{{mrr_all:.4f}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latency_table(results: list[dict]) -> str:
    wall_times = [r["timing"].get("wall_ms", 0) for r in results if "timing" in r]
    if not wall_times:
        return "% No timing data available"

    sorted_w = sorted(wall_times)
    p95_idx  = int(len(sorted_w) * 0.95)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{RAG Retrieval Latency — Per-Query Wall Time (ms)}",
        r"\label{tab:rag_latency}",
        r"\begin{tabular}{lc}",
        r"\hline",
        r"\textbf{Statistic} & \textbf{Latency (ms)} \\",
        r"\hline",
        f"Mean   & {statistics.mean(wall_times):.1f} \\\\",
        f"Median & {statistics.median(wall_times):.1f} \\\\",
        f"P95    & {sorted_w[p95_idx]:.1f} \\\\",
        f"Min    & {min(wall_times):.1f} \\\\",
        f"Max    & {max(wall_times):.1f} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="evaluation_results.json")
    parser.add_argument("--out", default="latex_tables.tex")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run evaluate_rag.py first.")
        exit(1)

    results = load_results(str(results_path))
    print(f"Loaded {len(results)} valid results.\n")

    t1 = overall_table(results)
    t2 = disease_table(results)
    t3 = latency_table(results)

    output = "\n\n% ─── Table 1: Overall metrics ───\n" + t1
    output += "\n\n% ─── Table 2: Per-disease breakdown ───\n" + t2
    output += "\n\n% ─── Table 3: Latency ───\n" + t3

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\nSaved → {args.out}")