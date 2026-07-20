"""
Agent-level thresholds and constants for CassavaCare-Agent (Phase 4, Part 1).

These values come from the cahier des charges (FR3, FR5, §6.3). Where the
cahier des charges is internally inconsistent, the discrepancy is noted —
resolve it once, then keep the report and this file in sync.
"""
import os
from dotenv import load_dotenv

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


load_dotenv()

# --- Part 2: OpenWeather -------------------------------------------------
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5/forecast"
WEATHER_REQUEST_TIMEOUT_SECONDS: float = 5.0
WEATHER_MAX_RETRIES: int = 3
WEATHER_RETRY_BACKOFF_SECONDS: float = 1.0
WEATHER_LOOKAHEAD_ENTRIES: int = 8       # 8 * 3h = 24h -> FR5's "12-24h" window
WEATHER_CACHE_TTL_SECONDS: int = 900     # 15 min; forecast itself refreshes ~every 3h

# --- Part 3: Gemini (LLM synthesis) --------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash"     # free-tier default as of mid-2026;
                                            # swap to "gemini-2.5-flash-lite"
                                            # if you hit free-tier rate limits
GEMINI_MAX_OUTPUT_TOKENS: int = 512        # ~220 words + margin
GEMINI_TEMPERATURE: float = 0.3            # low — factual report, not creative writing
GEMINI_MAX_RETRIES: int = 3
GEMINI_RETRY_BACKOFF_SECONDS: float = 2.0

# --- Part 3: FastAPI upload/job handling ----------------------------------
UPLOAD_DIR: str = "data/uploads"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}   # FR1: JPG, PNG, JPEG