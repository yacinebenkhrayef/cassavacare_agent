# scripts/export_graph_diagram.py
from src.agent.graph import agent_graph

png_bytes = agent_graph.get_graph().draw_mermaid_png()
with open("outputs/agent_graph.png", "wb") as f:
    f.write(png_bytes)
print("Saved outputs/agent_graph.png")