from src.agent.nodes import check_confidence, check_disease_status, decision_node
from src.agent.config import CONFIDENCE_THRESHOLD, RAIN_PROBABILITY_THRESHOLD, WIND_SPEED_THRESHOLD_KMH
from src.agent.nodes import check_weather_status, weather_fallback_node

def test_check_confidence_low():
    state = {"confidence": CONFIDENCE_THRESHOLD - 0.05}
    assert check_confidence(state) == "low_confidence"


def test_check_confidence_sufficient():
    state = {"confidence": CONFIDENCE_THRESHOLD + 0.05}
    assert check_confidence(state) == "sufficient_confidence"


def test_check_disease_status_healthy():
    assert check_disease_status({"pred_disease": "healthy"}) == "healthy"


def test_check_disease_status_diseased():
    assert check_disease_status({"pred_disease": "mosaic_disease"}) == "diseased"


def test_decision_defer_on_rain():
    state = {"weather": {"rain_probability": RAIN_PROBABILITY_THRESHOLD + 0.1, "wind_speed_kmh": 5.0}}
    out = decision_node(state)
    assert out["decision"] == "defer"


def test_decision_avoid_aerial_on_wind():
    state = {"weather": {"rain_probability": 0.05, "wind_speed_kmh": WIND_SPEED_THRESHOLD_KMH + 1}}
    out = decision_node(state)
    assert out["decision"] == "avoid_aerial"


def test_decision_apply_when_favorable():
    state = {"weather": {"rain_probability": 0.05, "wind_speed_kmh": 5.0}}
    out = decision_node(state)
    assert out["decision"] == "apply"



def test_check_weather_status_ok():
    assert check_weather_status({"weather": {"rain_probability": 0.1, "wind_speed_kmh": 5.0}}) == "ok"


def test_check_weather_status_unavailable():
    assert check_weather_status({"weather_error": "timeout"}) == "unavailable"


def test_weather_fallback_node_defers():
    state = {"weather_error": "OpenWeather unreachable", "trace": []}
    out = weather_fallback_node(state)
    assert out["decision"] == "defer"
    assert "précaution" in out["decision_reason"]

from unittest.mock import MagicMock, patch

from src.agent.nodes import synthesize_report_node
from src.llm_client import LLMUnavailableError


@patch("src.agent.nodes._llm_client")
def test_synthesize_report_node_success(mock_llm):
    mock_llm.generate_synthesis.return_value = "Your cassava plant has CMD..."
    state = {
        "pred_disease": "mosaic_disease",
        "pred_disease_short": "CMD",
        "confidence": 0.87,
        "rag_answer": "Apply mancozeb-based fungicide.",
        "rag_sources": [{"title": "FAO Technical Report 2024"}],
        "weather": {"rain_probability": 0.1, "wind_speed_kmh": 8.0, "forecast_hours": 24},
        "decision": "apply",
        "decision_reason": "Conditions favorables.",
        "trace": [],
    }
    out = synthesize_report_node(state)
    assert out["final_report"] == "Your cassava plant has CMD..."
    assert "Gemini" in out["trace"][-1]


@patch("src.agent.nodes._llm_client")
def test_synthesize_report_node_falls_back_on_llm_error(mock_llm):
    mock_llm.generate_synthesis.side_effect = LLMUnavailableError("Gemini unreachable")
    state = {
        "pred_disease": "mosaic_disease",
        "pred_disease_short": "CMD",
        "confidence": 0.87,
        "rag_answer": "Apply mancozeb-based fungicide.",
        "decision": "apply",
        "decision_reason": "Conditions favorables.",
        "trace": [],
    }
    out = synthesize_report_node(state)
    assert out["final_report"].startswith("[Fallback report")
    assert "indisponible" in out["trace"][-1]