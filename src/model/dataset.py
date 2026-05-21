import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
from db import get_db

def load_graph_from_db(repo_name):
    db = get_db()
    doc = db["repos"].find_one({"repo": repo_name})
    
    edges = doc["edges"]
    features = doc["features"]
    
    # step 1 - assign index to every function
    node_list = list(features.keys())
    node_index = {name: i for i, name in enumerate(node_list)}
    
    # step 2 - build feature matrix
    x = []
    for node in node_list:
        f = features[node]
        x.append([
            f["cyclomatic"],
            f["loc"],
            f["in_degree"],
            f["out_degree"]
        ])
    x = torch.tensor(x, dtype=torch.float)
    
    # step 3 - build edge index
    edge_src = []
    edge_dst = []
    for src, destinations in edges.items():
        if src in node_index:
            for dst in destinations:
                if dst in node_index:
                    edge_src.append(node_index[src])
                    edge_dst.append(node_index[dst])
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    
    return x, edge_index, node_list

if __name__ == "__main__":
    x, edge_index, node_list = load_graph_from_db("flask")
    print(f"nodes: {x.shape}")
    print(f"edges: {edge_index.shape}")
    print(f"first 3 functions: {node_list[:3]}")
    print(f"their features:\n{x[:3]}")