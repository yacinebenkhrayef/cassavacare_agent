"""
Phase 4, Part 4 — synthetic full-branch coverage for §6.3's five scenario
rows. Unlike scripts/validate_decision_accuracy.py (real E2E images -> the
actual "score global" you report), these tests monkeypatch exactly one
signal per scenario so every documented branch is exercised at least once,
regardless of the day's real weather.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.agent.nodes import initialize_agent_singletons
from src.agent.graph import agent_graph

SAMPLE_IMAGE = "data/samples/sample_leaf.jpg"  # any real, decodable leaf image

_results: dict[str, bool] = {}


@pytest.fixture(scope="module", autouse=True)
def _init():
    initialize_agent_singletons()


@pytest.mark.integration
@patch("src.agent.nodes.predict")
def test_low_confidence_requests_new_image(mock_predict):
    mock_predict.return_value = {
        "pred_class": 3, "pred_disease": "mosaic_disease",
        "confidence": 0.55, "probabilities": [0.1, 0.1, 0.1, 0.55, 0.15],
    }
    result = agent_graph.invoke({"image_path": SAMPLE_IMAGE, "location": "Tunis,TN"})
    ok = result.get("needs_new_image") is True
    _results["confidence < 0.7"] = ok
    assert ok


@pytest.mark.integration
@patch("src.agent.nodes._weather_client")
def test_high_rain_defers_treatment(mock_weather):
    mock_weather.get_forecast.return_value = {
        "rain_probability": 0.65, "wind_speed_kmh": 5.0, "forecast_hours": 24,
    }
    result = agent_graph.invoke({"image_path": SAMPLE_IMAGE, "location": "Tunis,TN"})
    ok = result.get("decision") == "defer"
    _results["rain > 30%"] = ok
    assert ok


@pytest.mark.integration
@patch("src.agent.nodes._weather_client")
def test_high_wind_avoids_aerial(mock_weather):
    mock_weather.get_forecast.return_value = {
        "rain_probability": 0.05, "wind_speed_kmh": 22.0, "forecast_hours": 24,
    }
    result = agent_graph.invoke({"image_path": SAMPLE_IMAGE, "location": "Tunis,TN"})
    ok = result.get("decision") == "avoid_aerial"
    _results["wind > 15 km/h"] = ok
    assert ok


@pytest.mark.integration
@patch("src.agent.nodes._weather_client")
def test_favorable_conditions_apply(mock_weather):
    mock_weather.get_forecast.return_value = {
        "rain_probability": 0.05, "wind_speed_kmh": 5.0, "forecast_hours": 24,
    }
    result = agent_graph.invoke({"image_path": SAMPLE_IMAGE, "location": "Tunis,TN"})
    ok = result.get("decision") == "apply"
    _results["conditions OK"] = ok
    assert ok


@pytest.mark.integration
@patch("src.agent.nodes._rag_client")
@patch("src.agent.nodes.predict")
def test_rare_disease_cbsd_returns_enough_sources(mock_predict, mock_rag_client):
    # Confirm pred_class=1 matches CBSD in Kaggle's mapping: 0=CBB, 1=CBSD, 2=CGM, 3=CMD, 4=Healthy
    mock_predict.return_value = {
        "pred_class": 1, "pred_disease": "brown_streak_disease",
        "confidence": 0.90, "probabilities": [0.02, 0.90, 0.02, 0.02, 0.04],
    }

    # Mock RAG client response with >= 2 sources (compatible with dict & dataclass formats)
    mock_rag_response = MagicMock()
    mock_rag_response.answer = "Use clean planting materials and rogue infected cassava plants."
    mock_rag_response.sources = [
        {"title": "FAO Cassava Disease Control Manual", "score": 0.89},
        {"title": "IITA Field Guide on CBSD Management", "score": 0.85},
    ]
    mock_rag_client.ask.return_value = mock_rag_response

    result = agent_graph.invoke({"image_path": SAMPLE_IMAGE, "location": "Tunis,TN"})
    ok = len(result.get("rag_sources", [])) >= 2
    _results["CBSD -> >=2 RAG sources"] = ok
    assert ok


def test_zzz_print_coverage_summary():
    """Named to sort last within this module (pytest's default alphabetical
    collection order) purely so it prints after the 5 scenario tests above
    have populated _results. Skip this file's tests individually with -k and
    this one will just report an empty summary — that's expected."""
    if not _results:
        pytest.skip("Run the full module, not this test alone.")
    correct = sum(1 for v in _results.values() if v)
    total = len(_results)
    print("\n--- §6.3 synthetic scenario coverage ---")
    for name, ok in _results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"Coverage: {correct}/{total} branch types exercised ({correct / total * 100:.0f}%)")