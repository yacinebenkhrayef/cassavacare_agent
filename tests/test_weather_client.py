import pytest
import responses
import requests

from src.weather_client import (
    OpenWeatherClient,
    WeatherAuthError,
    WeatherLocationNotFoundError,
    WeatherResponseError,
    WeatherTimeoutError,
)

BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"


def make_forecast_payload(pops, wind_speeds_ms):
    return {
        "cod": "200",
        "list": [
            {"pop": p, "wind": {"speed": w}}
            for p, w in zip(pops, wind_speeds_ms)
        ],
    }


@responses.activate
def test_get_forecast_success():
    responses.add(
        responses.GET, BASE_URL, json=make_forecast_payload(
            pops=[0.1, 0.2, 0.65, 0.3, 0.1, 0.0, 0.0, 0.0],
            wind_speeds_ms=[2.0, 3.0, 4.5, 2.0, 1.0, 1.0, 1.0, 1.0],
        ),
        status=200,
    )
    client = OpenWeatherClient(api_key="fake")
    result = client.get_forecast("Tunis,TN")
    assert result["rain_probability"] == 0.65
    assert result["wind_speed_kmh"] == pytest.approx(4.5 * 3.6, abs=0.1)
    assert result["forecast_hours"] == 24


@responses.activate
def test_get_forecast_missing_field_raises():
    responses.add(responses.GET, BASE_URL, json={"cod": "200", "list": [{"wind": {"speed": 1.0}}]}, status=200)
    client = OpenWeatherClient(api_key="fake")
    with pytest.raises(WeatherResponseError):
        client.get_forecast("Tunis,TN")


@responses.activate
def test_get_forecast_401_raises_auth_error():
    responses.add(responses.GET, BASE_URL, json={"cod": 401, "message": "Invalid API key"}, status=401)
    client = OpenWeatherClient(api_key="fake")
    with pytest.raises(WeatherAuthError):
        client.get_forecast("Tunis,TN")


@responses.activate
def test_get_forecast_404_raises_location_not_found():
    responses.add(responses.GET, BASE_URL, json={"cod": "404", "message": "city not found"}, status=404)
    client = OpenWeatherClient(api_key="fake")
    with pytest.raises(WeatherLocationNotFoundError):
        client.get_forecast("Notacityxyz")


@responses.activate
def test_get_forecast_timeout_raises_after_retries():
    for _ in range(3):  # WEATHER_MAX_RETRIES
        responses.add(responses.GET, BASE_URL, body=requests.exceptions.Timeout())
    client = OpenWeatherClient(api_key="fake")
    with pytest.raises(WeatherTimeoutError):
        client.get_forecast("Tunis,TN")


@responses.activate
def test_cache_avoids_second_http_call():
    responses.add(
        responses.GET, BASE_URL, json=make_forecast_payload(
            pops=[0.1] * 8, wind_speeds_ms=[1.0] * 8,
        ),
        status=200,
    )
    client = OpenWeatherClient(api_key="fake")
    client.get_forecast("Tunis,TN")
    client.get_forecast("Tunis,TN")  # should hit cache, not HTTP
    assert len(responses.calls) == 1