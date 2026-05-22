import ast
import sys
import os
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
from db import get_db
# debug - see what fields exist
print("Sample record keys:", list(dataset[0].keys()))
print("Sample record:", dataset[0])
def extract_functions_from_code(source_code):
    """
    Given raw Python source code as string,
    extract all function names.
    """
    try:
        tree = ast.parse(source_code)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions
    except Exception:
        return []

def load_cvefixes(limit=5000):
    """
    Loads CVEfixes from HuggingFace.
    Extracts Python functions labeled as vulnerable (risky=1)
    and fixed (risky=0).
    Saves to MongoDB.
    """
    print("loading CVEfixes dataset from HuggingFace...")
    dataset = load_dataset("hitoshura25/cvefixes", split="train")
    
    db = get_db()
    collection = db["labeled_functions"]
    
    inserted = 0
    skipped = 0

    for record in dataset:
        if inserted >= limit:
            break

        # only process Python files
        lang = record.get("programming_language", "")
        if lang.lower() != "python":
            skipped += 1
            continue

        vulnerable_code = record.get("vulnerable_code", "")
        fixed_code = record.get("fixed_code", "")

        if not vulnerable_code or not fixed_code:
            skipped += 1
            continue

        # extract functions from vulnerable version → risky = 1
        vuln_functions = extract_functions_from_code(vulnerable_code)
        for func_name in vuln_functions:
            collection.insert_one({
                "func_name": func_name,
                "source": "cvefixes",
                "label": 1,
                "code": vulnerable_code,
                "repo": record.get("repo_name", "unknown")
            })
            inserted += 1

        # extract functions from fixed version → risky = 0
        fixed_functions = extract_functions_from_code(fixed_code)
        for func_name in fixed_functions:
            collection.insert_one({
                "func_name": func_name,
                "source": "cvefixes",
                "label": 0,
                "code": fixed_code,
                "repo": record.get("repo_name", "unknown")
            })

    print(f"inserted {inserted} vulnerable functions")
    print(f"skipped {skipped} non-Python records")

if __name__ == "__main__":
    load_cvefixes(limit=5000)