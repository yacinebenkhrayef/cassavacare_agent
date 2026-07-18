"""
Manual smoke test: run the compiled graph on a real image + city, using
your real Phase 2 checkpoint, real Phase 3 RAG/Qdrant service, and now a
real OpenWeather call.
"""
import sys
from src.agent.graph import agent_graph

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/sample_leaf.jpg"
    location = sys.argv[2] if len(sys.argv) > 2 else "Tunis,TN"

    result = agent_graph.invoke({"image_path": image_path, "location": location})

    print("\n--- Trace (Affichage explicable) ---")
    for line in result.get("trace", []):
        print(line)

    print("\n--- Final report ---")
    print(result.get("final_report"))

if __name__ == "__main__":
    main()