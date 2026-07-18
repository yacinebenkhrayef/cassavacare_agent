"""
Shared state schema for the CassavaCare-Agent LangGraph workflow.

Using TypedDict (total=False) rather than Pydantic: LangGraph's node
functions each receive and return a partial dict of state updates, which
maps naturally onto a TypedDict with optional keys. Every node only needs
to declare the keys it actually sets.
"""
from typing import TypedDict, List, Dict, Optional, Literal


class AgentState(TypedDict, total=False):
    # ---- Input ----
    image_path: str
    location: str          # NEW (Part 2) — city name typed by the user, e.g. "Tunis,TN"

    # ---- Step 1: Diagnosis (EfficientNet + Grad-CAM) ----
    pred_class: int
    pred_disease: str          # snake_case label, e.g. "mosaic_disease"
    pred_disease_short: str    # short code, e.g. "CMD" (matches SHORT_NAMES)
    confidence: float
    probabilities: List[float]
    gradcam_path: str

    # ---- Step 2: Confidence gate ----
    needs_new_image: bool

    # ---- Step 3: RAG treatment retrieval ----
    rag_answer: str
    rag_sources: List[Dict]

    # ---- Step 4: Weather (STUB in Part 1 — real OpenWeather call in Part 2) ----
    weather: Dict[str, float]
    weather_error: Optional[str]   # NEW (Part 2) — set only if the API call failed

    # ---- Step 5: Decision ----
    decision: Literal["apply", "defer", "avoid_aerial", "no_action_needed", "pending"]
    decision_reason: str

    # ---- Step 6: Synthesis (STUB in Part 1 — real LLM call in Part 3) ----
    final_report: str

    # ---- Bookkeeping ----
    # Human-readable step log, matches the "Affichage explicable (obligatoire)"
    # format required in §4 of the cahier des charges (Étape 1 – ... Étape 2 – ...).
    trace: List[str]
    error: Optional[str]