"""
Agent-level thresholds and constants for CassavaCare-Agent (Phase 4, Part 1).

These values come from the cahier des charges (FR3, FR5, §6.3). Where the
cahier des charges is internally inconsistent, the discrepancy is noted —
resolve it once, then keep the report and this file in sync.
"""

# FR3 — Vérification confiance
CONFIDENCE_THRESHOLD: float = 0.70  # < 0.70 -> request a new image

# FR5 says "> 30 %" ; §6.3 KPI table says "> 50 % dans les 6h".
# Using FR5's value here — CHANGE THIS if you decide to standardize on 50%,
# and update the corresponding sentence in your report.
RAIN_PROBABILITY_THRESHOLD: float = 0.30

# FR5 — vitesse du vent
WIND_SPEED_THRESHOLD_KMH: float = 15.0

# Grad-CAM overlays output directory (created if missing)
GRADCAM_OUTPUT_DIR: str = "outputs/gradcam"

# Phase 2 checkpoint + config (adjust path to your actual checkpoint)
CHECKPOINT_PATH: str = "models/best_model_scripted_efficientnetv2_s.pt"
IMG_SIZE: int = 224