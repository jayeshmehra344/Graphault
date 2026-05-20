import ast
import os
import networkx as nx
import json 

def parse_file(filepath):
    with open(filepath, 'r') as f:
        source = f.read()
    tree = ast.parse(source)    
    return tree

def extract_functions(tree):
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            calls = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                   if isinstance(child.func, ast.Name):
                       calls.append(child.func.id)
            functions[func_name] = calls
    return functions 

def parse_repo(repo_path):
    all_functions = {}
    for root, dirs, files in os.walk(repo_path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root,filename)
                tree = parse_file(filepath)
                functions = extract_functions(tree)
                all_functions.update(functions)
    return all_functions     
    
    
def cyclomatic_complexity(tree, func_name):
    complexity = 1

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp)):
                    complexity += 1

    return complexity

def count_lines(tree, func_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.end_lineno - node.lineno + 1
    return 0

def analyze_repo(repo_path):
    all_functions = {}
    all_features = {}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                tree = parse_file(filepath)
                functions = extract_functions(tree)

                for func_name in functions:
                    cc = cyclomatic_complexity(tree, func_name)
                    loc = count_lines(tree, func_name)
                    all_features[func_name] = {
                        "file": filename,
                        "cyclomatic": cc,
                        "loc": loc,
                    }

                all_functions.update(functions)

    # THIS BLOCK must be outside the loop — after ALL files are processed
    graph = nx.DiGraph()
    for func, calls in all_functions.items():
        graph.add_node(func)
        for call in calls:
            if call in all_functions:
                graph.add_edge(func, call)

    for func in all_features:
        all_features[func]["in_degree"] = graph.in_degree(func)
        all_features[func]["out_degree"] = graph.out_degree(func)

    return all_functions, all_features

def save_features(all_functions, all_features, output_path):
    data = {
        "edges": all_functions,
        "features": all_features
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    functions, features = analyze_repo("../../data/sample_repo")
    
    for func, data in features.items():
        print(f"{func:<20} cc={data['cyclomatic']}  loc={data['loc']}  in={data['in_degree']}  out={data['out_degree']}")
    
    save_features(functions, features, "../../data/graph.json")