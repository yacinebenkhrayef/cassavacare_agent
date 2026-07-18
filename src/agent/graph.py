"""
Builds and compiles the CassavaCare-Agent LangGraph workflow (Phase 4, Part 1).
"""
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    predict_disease_node,
    check_confidence,
    request_new_image_node,
    retrieve_treatment_node,
    check_disease_status,
    weather_check_node,
    check_weather_status,     
    weather_fallback_node,    
    decision_node,
    synthesize_report_node,
)


def build_graph():
    graph = StateGraph(AgentState)

    # 1. Register Nodes
    graph.add_node("predict_disease", predict_disease_node)
    graph.add_node("request_new_image", request_new_image_node)
    graph.add_node("retrieve_treatment", retrieve_treatment_node)
    graph.add_node("weather_check", weather_check_node)
    graph.add_node("weather_fallback", weather_fallback_node)
    graph.add_node("decision", decision_node)
    graph.add_node("synthesize_report", synthesize_report_node)

    # 2. Define Flow (Edges)
    
    # --- STARTING POINT ---
    graph.add_edge(START, "predict_disease")

    # Step 2: Confidence Gate
    graph.add_conditional_edges(
        "predict_disease",
        check_confidence,
        {
            "low_confidence": "request_new_image",
            "sufficient_confidence": "retrieve_treatment",
        },
    )
    graph.add_edge("request_new_image", END)

    # Step 3: Healthy-leaf shortcut (skip weather + decision if healthy)
    graph.add_conditional_edges(
        "retrieve_treatment",
        check_disease_status,
        {
            "healthy": "synthesize_report",
            "diseased": "weather_check",
        },
    )

    # Step 4: Weather Validation Routing
    graph.add_conditional_edges(
        "weather_check",
        check_weather_status,
        {
            "ok": "decision",
            "unavailable": "weather_fallback",
        },
    )
    
    # Connect fallback path back into the workflow
    graph.add_edge("weather_fallback", "decision")

    # Final Execution Path
    graph.add_edge("decision", "synthesize_report")
    graph.add_edge("synthesize_report", END)

    return graph.compile()


# Module-level compiled graph
agent_graph = build_graph()