import sys
from src.agent.nodes import initialize_agent_singletons   # NEW
from src.agent.graph import agent_graph


def main():
    initialize_agent_singletons()   # NEW — singletons are now lazy
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