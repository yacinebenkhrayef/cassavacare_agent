from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(autouse=True)
def mock_singletons():
    # Skip loading the real checkpoint/Qdrant/weather/Gemini clients — the
    # graph itself is mocked per-test below, so nothing needs to be real.
    with patch("src.api.main.initialize_agent_singletons"):
        yield


def test_submit_rejects_bad_content_type():
    client = TestClient(app)
    resp = client.post(
        "/diagnose",
        files={"image": ("leaf.txt", b"not an image", "text/plain")},
        data={"location": "Tunis,TN"},
    )
    assert resp.status_code == 415


@patch("src.api.main.agent_graph")
def test_submit_and_poll_completed_job(mock_graph):
    mock_graph.invoke.return_value = {
        "pred_disease": "mosaic_disease",
        "pred_disease_short": "CMD",
        "confidence": 0.91,
        "gradcam_path": "outputs/gradcam/leaf_gradcam.png",
        "rag_sources": [{"title": "FAO Technical Report 2024"}],
        "weather": {"rain_probability": 0.1, "wind_speed_kmh": 8.0, "forecast_hours": 24},
        "decision": "apply",
        "decision_reason": "Conditions favorables.",
        "final_report": "The model detected cassava mosaic disease...",
        "trace": ["Étape 1 – ...", "Étape 6 – ..."],
    }
    client = TestClient(app)

    submit_resp = client.post(
        "/diagnose",
        files={"image": ("leaf.jpg", b"\xff\xd8\xff\xe0fake-jpeg-bytes", "image/jpeg")},
        data={"location": "Tunis,TN"},
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["job_id"]

    # TestClient runs the full ASGI lifecycle, including BackgroundTasks,
    # before returning the response — the job is already done by the time
    # submit_resp comes back, so this poll doesn't need a retry loop.
    status_resp = client.get(f"/diagnose/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == "completed"
    assert body["result"]["pred_disease_short"] == "CMD"


def test_poll_unknown_job_returns_404():
    client = TestClient(app)
    resp = client.get("/diagnose/does-not-exist")
    assert resp.status_code == 404