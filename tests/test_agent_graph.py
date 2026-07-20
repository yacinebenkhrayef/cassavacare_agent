import pytest
from src.agent.nodes import initialize_agent_singletons   # NEW
from src.agent.graph import agent_graph

SAMPLE_IMAGE = "data/samples/sample_leaf.jpg"


@pytest.fixture(scope="module", autouse=True)
def _init_singletons():
    initialize_agent_singletons()   # NEW


@pytest.mark.integration
def test_full_graph_runs_end_to_end():
    result = agent_graph.invoke({"image_path": SAMPLE_IMAGE, "location": "Tunis,TN"})
    assert "final_report" in result
    assert "trace" in result