"""Unit tests for src/llm_client.py — the Gemini API is mocked out entirely,
no network calls, no API key needed."""
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from src.llm_client import (
    GeminiClient,
    LLMAuthError,
    LLMRateLimitError,
    LLMResponseError,
)


def _mock_response(text="A short farmer-facing report.", finish_reason="STOP"):
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    response = MagicMock()
    response.candidates = [candidate]
    response.text = text
    return response


def test_missing_api_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        GeminiClient(api_key="")


@patch("src.llm_client.genai.Client")
def test_generate_synthesis_success(mock_client_cls):
    mock_client_cls.return_value.models.generate_content.return_value = _mock_response()
    client = GeminiClient(api_key="fake")
    assert client.generate_synthesis("some prompt") == "A short farmer-facing report."


@patch("src.llm_client.genai.Client")
def test_generate_synthesis_empty_response_raises(mock_client_cls):
    mock_client_cls.return_value.models.generate_content.return_value = _mock_response(text="")
    client = GeminiClient(api_key="fake")
    with pytest.raises(LLMResponseError):
        client.generate_synthesis("some prompt")


@patch("src.llm_client.genai.Client")
def test_generate_synthesis_401_raises_auth_error(mock_client_cls):
    # NOTE: ClientError's exact response_json parsing may differ slightly by
    # SDK version — if this construction errors on your installed version,
    # check `google/genai/errors.py` and adjust the payload shape.
    error = genai_errors.ClientError(
        code=401, response_json={"error": {"message": "Invalid API key", "status": "UNAUTHENTICATED"}}
    )
    mock_client_cls.return_value.models.generate_content.side_effect = error
    client = GeminiClient(api_key="fake")
    with pytest.raises(LLMAuthError):
        client.generate_synthesis("some prompt")


@patch("src.llm_client.time.sleep", return_value=None)  # skip real backoff delays
@patch("src.llm_client.genai.Client")
def test_generate_synthesis_429_raises_rate_limit_after_retries(mock_client_cls, mock_sleep):
    error = genai_errors.ClientError(
        code=429, response_json={"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )
    mock_client_cls.return_value.models.generate_content.side_effect = error
    client = GeminiClient(api_key="fake")
    with pytest.raises(LLMRateLimitError):
        client.generate_synthesis("some prompt")