import ast
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

if __name__ == "__main__":
    tree = parse_file("data/sample_repo/example.py")
    result = extract_functions(tree)
    print(result)