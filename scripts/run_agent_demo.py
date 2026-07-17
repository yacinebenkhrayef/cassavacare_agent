"""
Manual smoke test: run the compiled graph on a real image using your real
Phase 2 checkpoint and real Phase 3 RAG/Qdrant service.
"""
import sys
from src.agent.graph import agent_graph

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/sample_leaf.jpg"

    result = agent_graph.invoke({"image_path": image_path})

    print("\n--- Trace (Affichage explicable) ---")
    for line in result.get("trace", []):
        print(line)

    print("\n--- Final report ---")
    print(result.get("final_report"))

    print("\n--- Raw state keys ---")
    print(sorted(result.keys()))

if __name__ == "__main__":
    main()