import networkx as nx
import matplotlib.pyplot as plt
from parse import parse_repo

def build_graph(all_functions):
    graph = nx.DiGraph()

    for func, calls in all_functions.items():
        graph.add_node(func)
        for call in calls:
            if call in all_functions:
                graph.add_edge(func, call)

    return graph

def visualize(graph):
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(graph, seed=42, k=3)
    
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color="#4f86f7")
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="white")
    
    plt.title("CodeSense — Function Call Graph")
    plt.axis("off")
    plt.margins(0.2)
    plt.tight_layout()
    plt.savefig("graph.png", dpi=150)
    print("saved to graph.png")

if __name__ == "__main__":
    all_functions = parse_repo("../../data/sample_repo")
    graph = build_graph(all_functions)
    print("Nodes:", list(graph.nodes()))
    print("Edges:", list(graph.edges()))
    visualize(graph)