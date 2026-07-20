"""
LangGraph node functions for CassavaCare-Agent (Phase 4, Part 3).

Model, Grad-CAM wrapper, RAG, Weather, and LLM clients are initialized lazily
via `initialize_agent_singletons()` during FastAPI app startup or test setups.
"""
import torch

from src.model import load_model
from src.gradcam import GradCAMWrapper
from src.inference import predict
from src.config import CFG, LABEL_MAP, SHORT_NAMES
from src.weather_client import OpenWeatherClient, WeatherAPIError
from src.llm_client import GeminiClient, LLMAPIError
from src.agent.prompts import build_diagnosis_prompt, build_healthy_prompt

# TODO: adjust this import to match your actual Phase 3 project structure.
from api.client import CassavaRAGClient

from src.agent.config import (
    CONFIDENCE_THRESHOLD,
    RAIN_PROBABILITY_THRESHOLD,
    WIND_SPEED_THRESHOLD_KMH,
    GRADCAM_OUTPUT_DIR,
    CHECKPOINT_PATH,
    IMG_SIZE,
)
from src.agent.state import AgentState
from src.agent.utils import save_gradcam_overlay, disease_query_text

# ---------------------------------------------------------------------------
# Module-level singletons — now lazily created by initialize_agent_singletons(),
# called once from FastAPI's lifespan handler (src/api/main.py) instead of at
# module-import time. All node functions below read these same names as
# module globals — nothing else in this file changes.
# ---------------------------------------------------------------------------
_device = None
_model = None
_cam_wrapper = None
_rag_client = None
_weather_client = None
_llm_client = None


def initialize_agent_singletons() -> None:
    """Idempotent — safe to call more than once (e.g. once per test module).
    Call this exactly once per process, before the first graph invocation."""
    global _device, _model, _cam_wrapper, _rag_client, _weather_client, _llm_client
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = load_model(CHECKPOINT_PATH, CFG, device=_device)
    _cam_wrapper = GradCAMWrapper(_model, device=_device, img_size=IMG_SIZE)
    _rag_client = CassavaRAGClient()
    _weather_client = OpenWeatherClient()
    _llm_client = GeminiClient()


# ---------------------------------------------------------------------------
# Step 1 — Diagnosis (EfficientNet + Grad-CAM)
# ---------------------------------------------------------------------------
def predict_disease_node(state: AgentState) -> dict:
    image_path = state["image_path"]

    result = predict(
        _model, image_path, _device,
        img_size=IMG_SIZE, label_map={str(k): v for k, v in LABEL_MAP.items()},
    )
    pred_class = result["pred_class"]

    # Pass pred_class explicitly so the heatmap targets the same class predict()
    # reported (model is in eval mode, so this is deterministic, but being
    # explicit avoids any ambiguity between the two forward passes).
    _, _, overlay = _cam_wrapper.generate_heatmap(image_path, target_class=pred_class)
    gradcam_path = save_gradcam_overlay(overlay, image_path, GRADCAM_OUTPUT_DIR)

    trace_line = (
        f"Étape 1 – Maladie détectée : {SHORT_NAMES[pred_class]} "
        f"({result['pred_disease']}) – Confiance : {result['confidence']:.2f}"
    )

    return {
        "pred_class": pred_class,
        "pred_disease": result["pred_disease"],
        "pred_disease_short": SHORT_NAMES[pred_class],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "gradcam_path": gradcam_path,
        "trace": state.get("trace", []) + [trace_line],
    }


# ---------------------------------------------------------------------------
# Step 2 — Confidence gate (conditional edge, no state mutation)
# ---------------------------------------------------------------------------
def check_confidence(state: AgentState) -> str:
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return "sufficient_confidence"


def request_new_image_node(state: AgentState) -> dict:
    trace_line = (
        f"Étape 2 – Confiance insuffisante ({state['confidence']:.2f} < "
        f"{CONFIDENCE_THRESHOLD}) – Nouvelle image demandée."
    )
    return {
        "needs_new_image": True,
        "final_report": "Confiance insuffisante : merci de fournir une nouvelle image plus nette.",
        "trace": state.get("trace", []) + [trace_line],
    }


# ---------------------------------------------------------------------------
# Step 3 — RAG treatment retrieval
# ---------------------------------------------------------------------------
def retrieve_treatment_node(state: AgentState) -> dict:
    query = disease_query_text(state["pred_disease"])

    rag_result = _rag_client.ask(query)
    answer = rag_result.get("answer", "")
    sources = rag_result.get("sources", [])

    trace_line = (
        f"Étape 3 – Traitement trouvé : {answer[:120]}"
        f"{'...' if len(answer) > 120 else ''} – "
        f"{len(sources)} source(s) récupérée(s)."
    )

    return {
        "rag_answer": answer,
        "rag_sources": sources,
        "trace": state.get("trace", []) + [trace_line],
    }


# ---------------------------------------------------------------------------
# Routing after RAG: skip weather/decision entirely for healthy leaves
# ---------------------------------------------------------------------------
def check_disease_status(state: AgentState) -> str:
    if state["pred_disease"] == "healthy":
        return "healthy"
    return "diseased"


# ---------------------------------------------------------------------------
# Step 4 — Weather check
# ---------------------------------------------------------------------------
def weather_check_node(state: AgentState) -> dict:
    city = state["location"]
    try:
        weather = _weather_client.get_forecast(city)
    except WeatherAPIError as exc:
        trace_line = f"Étape 4 – Météo indisponible ({city}) : {exc}"
        return {
            "weather_error": str(exc),
            "trace": state.get("trace", []) + [trace_line],
        }

    trace_line = (
        f"Étape 4 – Météo locale ({city}) : pluie {weather['rain_probability']*100:.0f}% "
        f"– Vent {weather['wind_speed_kmh']:.1f} km/h (fenêtre {weather['forecast_hours']}h)"
    )
    return {
        "weather": weather,
        "trace": state.get("trace", []) + [trace_line],
    }


# ---------------------------------------------------------------------------
# Routing after weather
# ---------------------------------------------------------------------------
def check_weather_status(state: AgentState) -> str:
    return "unavailable" if state.get("weather_error") else "ok"


def weather_fallback_node(state: AgentState) -> dict:
    """
    Safety-first fallback when the weather API is unreachable or
    misconfigured. Agronomic reasoning: it's better to defer a treatment
    unnecessarily than to apply one during conditions we couldn't verify.
    """
    reason = (
        f"Données météo indisponibles ({state.get('weather_error', 'erreur inconnue')}) "
        f"— traitement reporté par précaution."
    )
    trace_line = f"Étape 5 – Décision (fallback) : {reason}"
    return {
        "decision": "defer",
        "decision_reason": reason,
        "trace": state.get("trace", []) + [trace_line],
    }


# ---------------------------------------------------------------------------
# Step 5 — Decision
# ---------------------------------------------------------------------------
def decision_node(state: AgentState) -> dict:
    weather = state["weather"]
    rain = weather["rain_probability"]
    wind = weather["wind_speed_kmh"]

    if rain > RAIN_PROBABILITY_THRESHOLD:
        decision = "defer"
        reason = (
            f"Probabilité de pluie {rain*100:.0f}% > seuil "
            f"{RAIN_PROBABILITY_THRESHOLD*100:.0f}% — traitement reporté."
        )
    elif wind > WIND_SPEED_THRESHOLD_KMH:
        decision = "avoid_aerial"
        reason = (
            f"Vent {wind:.0f} km/h > seuil {WIND_SPEED_THRESHOLD_KMH:.0f} km/h — "
            f"pulvérisation aérienne évitée (application au sol possible)."
        )
    else:
        decision = "apply"
        reason = "Conditions météo favorables — traitement applicable immédiatement."

    trace_line = f"Étape 5 – Décision : {reason}"

    return {
        "decision": decision,
        "decision_reason": reason,
        "trace": state.get("trace", []) + [trace_line],
    }


# ---------------------------------------------------------------------------
# Step 6 — Synthesis (REAL — Part 3). Falls back to a templated report if
# Gemini is unavailable for any reason, rather than letting the whole graph
# invocation fail — a farmer should still get *a* report, even a plain one.
# ---------------------------------------------------------------------------
def synthesize_report_node(state: AgentState) -> dict:
    if state["pred_disease"] == "healthy":
        prompt = build_healthy_prompt(state)
    else:
        prompt = build_diagnosis_prompt(state)

    try:
        report = _llm_client.generate_synthesis(prompt)
        trace_line = "Étape 6 – Rapport de synthèse généré via Gemini."
    except LLMAPIError as exc:
        report = _fallback_report(state, str(exc))
        trace_line = f"Étape 6 – Gemini indisponible ({exc}) – repli sur gabarit statique."

    default_decision = "no_action_needed" if state["pred_disease"] == "healthy" else "pending"

    return {
        "final_report": report,
        "decision": state.get("decision", default_decision),
        "trace": state.get("trace", []) + [trace_line],
    }


def _fallback_report(state: AgentState, error_message: str) -> str:
    """Same content as the Part 1 stub, in English, plus an explicit flag so
    this is never mistaken for a real LLM synthesis in the dashboard (Phase 5)
    or in a §6.4 UX evaluation."""
    if state["pred_disease"] == "healthy":
        return (
            f"[Fallback report — Gemini unavailable: {error_message}]\n"
            f"Diagnosis: healthy plant (confidence {state['confidence']:.2f}). "
            f"No action required."
        )
    return (
        f"[Fallback report — Gemini unavailable: {error_message}]\n"
        f"Diagnosis: {state['pred_disease_short']} (confidence {state['confidence']:.2f}).\n"
        f"Recommended treatment: {state.get('rag_answer', 'N/A')}\n"
        f"Weather-based decision: {state.get('decision_reason', 'N/A')}"
    )