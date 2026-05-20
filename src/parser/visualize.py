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
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42, k=2)

    nx.draw_networkx_nodes(graph, pos, node_size=1500, node_color="#4f86f7", alpha=0.9)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=15, edge_color="#aaaaaa")
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white", font_weight="bold")

    plt.title("CodeSense — Function Call Graph", fontsize=14)
    plt.axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("graph.png", dpi=150, bbox_inches="tight")
    print("saved to graph.png")

if __name__ == "__main__":
    all_functions = parse_repo("../../data/sample_repo")
    graph = build_graph(all_functions)
    print("Nodes:", list(graph.nodes()))
    print("Edges:", list(graph.edges()))
    visualize(graph)