import ast
import os
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
    

if __name__ == "__main__":
    result = parse_repo("data/sample_repo")
    print(result)