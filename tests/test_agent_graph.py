import pytest
from src.agent.graph import agent_graph

SAMPLE_IMAGE = "data/samples/sample_leaf.jpg"  # adjust to a real path you have


@pytest.mark.integration
def test_full_graph_runs_end_to_end():
    result = agent_graph.invoke({"image_path": SAMPLE_IMAGE})
    assert "final_report" in result
    assert "trace" in result
    assert len(result["trace"]) >= 1
    # Either the confidence gate fired, or the full pipeline ran through
    assert result.get("needs_new_image") is True or "decision" in result or result.get("pred_disease") == "healthy"