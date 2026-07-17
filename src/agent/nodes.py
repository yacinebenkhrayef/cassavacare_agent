"""
LangGraph node functions for CassavaCare-Agent (Phase 4, Part 1).

Model, Grad-CAM wrapper, and RAG client are loaded once at module import
time (singleton pattern). In Part 3, when this graph is wrapped by FastAPI,
this loading moves into a startup event instead of module import — noted
inline below.
"""
import torch

from src.model import load_model
from src.gradcam import GradCAMWrapper
from src.inference import predict
from src.config import CFG, LABEL_MAP, SHORT_NAMES

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
# Module-level singletons — loaded once, reused across every graph invocation.
# In Part 3 (FastAPI), move this block into an `@app.on_event("startup")`
# handler and store the objects on `app.state` instead of module globals.
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = load_model(CHECKPOINT_PATH, CFG, device=_device)
_cam_wrapper = GradCAMWrapper(_model, device=_device, img_size=IMG_SIZE)
_rag_client = CassavaRAGClient()  # TODO: pass constructor args your Phase 3 client needs


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

    # NOTE: verify this against your actual Phase 3 CassavaRAGClient.ask()
    # return shape — adjust the two field names below if they differ.
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
# Step 4 — Weather (STUB — replaced by a real OpenWeather call in Part 2)
# ---------------------------------------------------------------------------
def weather_check_node(state: AgentState) -> dict:
    """
    STUB for Phase 4, Part 2.
    Returns fixed mock values so the decision node (Step 5, already real)
    can be built and tested now. Part 2 replaces the body of this function
    with a real OpenWeather API call — the return shape (rain_probability,
    wind_speed_kmh, forecast_hours) must stay the same so decision_node
    doesn't need to change.
    """
    mock_weather = {
        "rain_probability": 0.10,   # TODO Part 2: real OpenWeather value
        "wind_speed_kmh": 8.0,      # TODO Part 2: real OpenWeather value
        "forecast_hours": 12,
    }
    trace_line = (
        f"Étape 4 – Météo locale (MOCK) : pluie {mock_weather['rain_probability']*100:.0f}% "
        f"– Vent {mock_weather['wind_speed_kmh']} km/h [STUB — Partie 2]"
    )
    return {
        "weather": mock_weather,
        "trace": state.get("trace", []) + [trace_line],
    }


# ---------------------------------------------------------------------------
# Step 5 — Decision (fully implemented — business rules from FR5 / §6.3)
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
# Step 6 — Synthesis (STUB — replaced by a real Gemini call in Part 3)
# ---------------------------------------------------------------------------
def synthesize_report_node(state: AgentState) -> dict:
    """
    STUB for Phase 4, Part 3.
    Assembles the "Affichage explicable" trace into a plain-text report
    without any LLM call. Part 3 replaces this with a real Gemini prompt
    that takes the same state fields and produces a natural-language
    synthesis instead of this templated join.
    """
    if state["pred_disease"] == "healthy":
        report = (
            f"Diagnostic : plante saine (confiance {state['confidence']:.2f}). "
            f"Aucune action requise."
        )
        decision_summary = "no_action_needed"
    else:
        report = (
            f"Diagnostic : {state['pred_disease_short']} "
            f"(confiance {state['confidence']:.2f}).\n"
            f"Traitement recommandé : {state.get('rag_answer', 'N/A')}\n"
            f"Décision météo : {state.get('decision_reason', 'N/A')} [STUB — Partie 3 pour la synthèse LLM]"
        )
        decision_summary = state.get("decision", "pending")

    trace_line = "Étape 6 – Rapport généré (gabarit statique, LLM en Partie 3)."

    return {
        "final_report": report,
        "decision": state.get("decision", decision_summary),
        "trace": state.get("trace", []) + [trace_line],
    }