"""
Gemini client for CassavaCare-Agent report synthesis (Phase 4, Part 3).

Uses the unified `google-genai` SDK (NOT the deprecated `google-generativeai`
package). Always returns plain report text or raises an LLMAPIError
subclass — nodes.py decides what "no report" means for the graph.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from src.agent.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)

# Persona + hard rules for the model. Kept here (not in prompts.py) because
# this is the one place that talks to the Gemini config API — a wording
# change to the report style never needs to touch prompts.py, and a change
# to *what data* goes in never needs to touch this file.
SYSTEM_INSTRUCTION = (
    "You are the reporting component of CassavaCare-Agent, an agronomic "
    "decision support tool for cassava leaf disease. You are given "
    "structured, already-verified data produced upstream by a trained "
    "image classifier, a document retrieval system, a weather API, and a "
    "rule-based decision engine — your only job is to turn that data into "
    "a clear, actionable report for a smallholder farmer.\n\n"
    "Rules:\n"
    "- Do not invent facts: no disease names, treatments, sources, or "
    "weather values beyond what is given to you in the message.\n"
    "- Do not second-guess the diagnosis, the retrieved treatment, or the "
    "decision — present them, don't re-derive them.\n"
    "- Write in plain English, 120-220 words, no markdown headers, no "
    "bullet symbols — short paragraphs a farmer can read on a phone screen.\n"
    "- Structure: (1) what was diagnosed and the model's confidence, "
    "(2) the recommended treatment and where it comes from, (3) the timing "
    "decision and the weather reasoning behind it (or, if weather data was "
    "unavailable, say so plainly and explain the precautionary default)."
)


class LLMAPIError(Exception):
    """Base class for all Gemini-client errors."""


class LLMAuthError(LLMAPIError):
    """401/403 — invalid, missing, or unauthorized API key. Not retried."""


class LLMRateLimitError(LLMAPIError):
    """429 — free-tier quota exceeded, even after retries."""


class LLMResponseError(LLMAPIError):
    """200 OK but no usable text (safety block, empty candidates, bad request)."""


class LLMUnavailableError(LLMAPIError):
    """5xx or a network-level failure, even after retries."""


class GeminiClient:
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        if not api_key:
            raise LLMAuthError("GEMINI_API_KEY is empty — set it in your .env file.")
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def generate_synthesis(self, prompt: str) -> str:
        """Returns the synthesized report text, or raises an LLMAPIError
        subclass. Retries on 429/5xx with linear backoff; auth errors and
        bad requests are not retried since retrying won't fix them."""
        last_exc: Optional[Exception] = None
        last_was_rate_limit = False

        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        temperature=GEMINI_TEMPERATURE,
                    ),
                )
            except genai_errors.ClientError as exc:
                if exc.code in (401, 403):
                    raise LLMAuthError(f"Gemini rejected the API key: {exc.message}") from exc
                if exc.code == 429:
                    last_exc, last_was_rate_limit = exc, True
                    logger.warning(
                        "Gemini rate-limited (attempt %d/%d): %s",
                        attempt, GEMINI_MAX_RETRIES, exc.message,
                    )
                    time.sleep(GEMINI_RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise LLMResponseError(
                    f"Gemini rejected the request ({exc.code}): {exc.message}"
                ) from exc
            except genai_errors.ServerError as exc:
                last_exc, last_was_rate_limit = exc, False
                logger.warning(
                    "Gemini server error (attempt %d/%d): %s",
                    attempt, GEMINI_MAX_RETRIES, exc.message,
                )
                time.sleep(GEMINI_RETRY_BACKOFF_SECONDS * attempt)
                continue
            except Exception as exc:  # network-level failures from the underlying transport
                last_exc, last_was_rate_limit = exc, False
                logger.warning(
                    "Gemini unreachable (attempt %d/%d): %s",
                    attempt, GEMINI_MAX_RETRIES, exc,
                )
                time.sleep(GEMINI_RETRY_BACKOFF_SECONDS * attempt)
                continue

            return self._extract_text(response)

        if last_was_rate_limit:
            raise LLMRateLimitError(
                f"Gemini rate limit exceeded after {GEMINI_MAX_RETRIES} attempts: {last_exc}"
            )
        raise LLMUnavailableError(
            f"Gemini unreachable after {GEMINI_MAX_RETRIES} attempts: {last_exc}"
        )

    @staticmethod
    def _extract_text(response) -> str:
        if not response.candidates:
            raise LLMResponseError("Gemini returned no candidates (likely a safety block).")
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None and str(finish_reason).split(".")[-1] != "STOP":
            raise LLMResponseError(f"Gemini stopped early: {finish_reason}")
        text = (response.text or "").strip()
        if not text:
            raise LLMResponseError("Gemini returned an empty response.")
        return text