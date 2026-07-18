"""
OpenWeather API client for CassavaCare-Agent (Phase 4, Part 2).

Wraps the free-tier "5 Day / 3-Hour Forecast" endpoint (deliberately NOT
One Call 3.0, which requires a card on file even on its free quota).
Always returns the same three-key dict — rain_probability, wind_speed_kmh,
forecast_hours — or raises a WeatherAPIError subclass. nodes.py decides
what "no weather data" means for the graph; this client just fetches
and validates.
"""
from __future__ import annotations

import time
import logging
from typing import Optional

import requests
from cachetools import TTLCache

from src.agent.config import (
    OPENWEATHER_API_KEY,
    OPENWEATHER_BASE_URL,
    WEATHER_REQUEST_TIMEOUT_SECONDS,
    WEATHER_LOOKAHEAD_ENTRIES,
    WEATHER_CACHE_TTL_SECONDS,
    WEATHER_MAX_RETRIES,
    WEATHER_RETRY_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)


class WeatherAPIError(Exception):
    """Base class for all weather-client errors."""


class WeatherAuthError(WeatherAPIError):
    """401 — invalid, missing, or not-yet-activated API key."""


class WeatherLocationNotFoundError(WeatherAPIError):
    """404 — OpenWeather doesn't recognize the city string."""


class WeatherTimeoutError(WeatherAPIError):
    """Unreachable after retries (timeouts / connection errors)."""


class WeatherResponseError(WeatherAPIError):
    """200 OK but the JSON body is missing fields we depend on."""


class OpenWeatherClient:
    def __init__(self, api_key: str = OPENWEATHER_API_KEY,
                 base_url: str = OPENWEATHER_BASE_URL,
                 timeout: float = WEATHER_REQUEST_TIMEOUT_SECONDS):
        if not api_key:
            raise WeatherAuthError(
                "OPENWEATHER_API_KEY is empty — set it in your .env file."
            )
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._cache: TTLCache = TTLCache(maxsize=64, ttl=WEATHER_CACHE_TTL_SECONDS)

    def get_forecast(self, city: str) -> dict:
        """Return {"rain_probability": float 0-1, "wind_speed_kmh": float,
        "forecast_hours": int}, worst-case over the next
        WEATHER_LOOKAHEAD_ENTRIES * 3 hours.
        """
        cache_key = city.strip().lower()
        if cache_key in self._cache:
            logger.info("Weather cache hit for %s", city)
            return self._cache[cache_key]

        payload = self._fetch_with_retries(city)
        parsed = self._parse_and_validate(payload)
        self._cache[cache_key] = parsed
        return parsed

    def _fetch_with_retries(self, city: str) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, WEATHER_MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    self._base_url,
                    params={"q": city, "appid": self._api_key, "units": "metric"},
                    timeout=self._timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                logger.warning(
                    "Weather request failed (attempt %d/%d): %s",
                    attempt, WEATHER_MAX_RETRIES, exc,
                )
                time.sleep(WEATHER_RETRY_BACKOFF_SECONDS * attempt)
                continue

            if resp.status_code == 401:
                raise WeatherAuthError(
                    "OpenWeather rejected the API key (401). If you just "
                    "created it, it can take up to 2 hours to activate — "
                    "otherwise check OPENWEATHER_API_KEY in .env."
                )
            if resp.status_code == 404:
                raise WeatherLocationNotFoundError(
                    f"OpenWeather doesn't recognize the city '{city}'. Try "
                    f"'{city},<ISO country code>' (e.g. 'Tunis,TN')."
                )
            if resp.status_code >= 500:
                last_exc = WeatherAPIError(f"OpenWeather server error {resp.status_code}")
                time.sleep(WEATHER_RETRY_BACKOFF_SECONDS * attempt)
                continue

            resp.raise_for_status()  # anything else unexpected -> requests.HTTPError
            return resp.json()

        raise WeatherTimeoutError(
            f"Weather API unreachable after {WEATHER_MAX_RETRIES} attempts: {last_exc}"
        )

    def _parse_and_validate(self, payload: dict) -> dict:
        entries = payload.get("list")
        if not entries:
            raise WeatherResponseError("Response has no 'list' entries (empty forecast).")

        window = entries[:WEATHER_LOOKAHEAD_ENTRIES]
        try:
            pops = [e["pop"] for e in window]
            winds_ms = [e["wind"]["speed"] for e in window]
        except KeyError as exc:
            raise WeatherResponseError(f"Forecast entry missing expected field: {exc}")

        rain_probability = max(pops)
        # OpenWeather returns wind speed in m/s even with units=metric —
        # only temperature switches units with that param. Convert explicitly.
        wind_speed_kmh = max(winds_ms) * 3.6

        return {
            "rain_probability": round(rain_probability, 2),
            "wind_speed_kmh": round(wind_speed_kmh, 1),
            "forecast_hours": WEATHER_LOOKAHEAD_ENTRIES * 3,
        }