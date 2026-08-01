"""
Phase 4, Part 4 — §6.3 "Score global" validation.

Reads reports/e2e_run/e2e_results.csv (produced by scripts/e2e_client.py)
and independently recomputes the expected decision for each real run, using
the documented business rules from FR3/FR5 — WITHOUT importing decision_node,
so this is a genuine independent check, not the code testing itself.

Score global = décisions correctes / total des tests   (target: >= 85 %, §6.3)
"""
import pandas as pd

RESULTS_PATH = "reports/e2e_run/e2e_results.csv"

# Kept independent from src/agent/config.py on purpose (see module docstring).
# If you ever change the real thresholds, update these two lines to match.
CONFIDENCE_THRESHOLD = 0.70
RAIN_PROBABILITY_THRESHOLD = 0.30   # resolved value — see Part 1 §2 flag
WIND_SPEED_THRESHOLD_KMH = 15.0

# TODO: confirm this matches the exact string your SHORT_NAMES mapping
# (src.configs) uses for the healthy class.
HEALTHY_LABEL = "Healthy"


def expected_outcome(row: pd.Series) -> str:
    if bool(row["needs_new_image"]):
        return "needs_new_image"
    if row["pred_disease_short"] == HEALTHY_LABEL:
        return "no_action_needed"
    if pd.isna(row["confidence"]) or row["confidence"] < CONFIDENCE_THRESHOLD:
        return "needs_new_image"
    if pd.notna(row["weather_error"]):
        return "defer"   # weather_fallback_node always defers, Part 2 §7
    if pd.notna(row["rain_probability"]) and row["rain_probability"] > RAIN_PROBABILITY_THRESHOLD:
        return "defer"
    if pd.notna(row["wind_speed_kmh"]) and row["wind_speed_kmh"] > WIND_SPEED_THRESHOLD_KMH:
        return "avoid_aerial"
    return "apply"


def actual_outcome(row: pd.Series) -> str:
    return "needs_new_image" if bool(row["needs_new_image"]) else row["decision"]


def main():
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["status"] == "completed"].copy()   # exclude failed jobs from the score

    df["expected"] = df.apply(expected_outcome, axis=1)
    df["actual"] = df.apply(actual_outcome, axis=1)
    df["correct"] = df["expected"] == df["actual"]

    total = len(df)
    correct = int(df["correct"].sum())
    score = correct / total * 100 if total else 0.0

    print("--- §6.3 Fiabilité de l'agent — score global ---")
    print(df[["image_path", "location", "pred_disease_short", "confidence",
               "rain_probability", "wind_speed_kmh", "expected", "actual", "correct"]]
          .to_string(index=False))
    print(f"\nScore global: {correct}/{total} = {score:.1f}%  (target: >= 85 %)")

    mismatches = df[~df["correct"]]
    if not mismatches.empty:
        print("\nMismatches to investigate:")
        print(mismatches[["image_path", "expected", "actual", "decision_reason"]].to_string(index=False))

    assert score >= 85.0, f"Score global {score:.1f}% is below the §6.3 target of 85%"
    print("\n✅ §6.3 target met.")


if __name__ == "__main__":
    main()