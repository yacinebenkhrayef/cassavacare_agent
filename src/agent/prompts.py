"""Builds the *data* portion of the Gemini prompt (Phase 4, Part 3).

The persona and formatting rules live in src/llm_client.py's
SYSTEM_INSTRUCTION — this file only turns verified AgentState fields into
plain text the model can read.
"""
from src.agent.state import AgentState


def build_diagnosis_prompt(state: AgentState) -> str:
    sources = ", ".join(
        s.get("title") or s.get("source", "unknown source")
        for s in state.get("rag_sources", [])
    ) or "no sources retrieved"

    if state.get("weather_error"):
        weather_block = (
            f"Weather data: UNAVAILABLE ({state['weather_error']}). "
            f"The decision below was taken as a precaution because of this."
        )
    else:
        weather = state.get("weather", {})
        weather_block = (
            f"Weather forecast (next {weather.get('forecast_hours', '?')}h): "
            f"{weather.get('rain_probability', 0) * 100:.0f}% rain probability, "
            f"{weather.get('wind_speed_kmh', 0):.1f} km/h wind."
        )

    return (
        f"Diagnosis: {state['pred_disease_short']} ({state['pred_disease']})\n"
        f"Model confidence: {state['confidence']:.2f}\n"
        f"Retrieved treatment guidance: {state.get('rag_answer', 'N/A')}\n"
        f"Treatment sources: {sources}\n"
        f"{weather_block}\n"
        f"Decision: {state.get('decision', 'pending')} — "
        f"{state.get('decision_reason', 'N/A')}\n\n"
        f"Write the farmer-facing report now."
    )


def build_healthy_prompt(state: AgentState) -> str:
    return (
        f"Diagnosis: healthy cassava leaf, no disease detected.\n"
        f"Model confidence: {state['confidence']:.2f}\n\n"
        f"Write a short (2-4 sentence) reassuring report confirming the "
        f"plant looks healthy and no treatment is needed. Do not mention "
        f"weather or retrieved documents — they were not queried for a "
        f"healthy leaf (see cahier des charges FR3 rationale)."
    )