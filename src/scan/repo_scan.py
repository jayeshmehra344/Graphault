"""
src/scan/repo_scan.py
======================
Scan a local Python repo/folder and risk-score every function with the
trained CodeRiskGNN model — fully in-process, no MongoDB, no HTTP.

Pipeline (mirrors src/api/app.py's /predict, minus the network hop):
  1. Walk the repo, collect .py files (skipping venv/, node_modules/, .git/, __pycache__/)
  2. For each file: parse with src/parser/parse.py's parse_file (reuse, no new parser)
  3. For each FunctionDef/AsyncFunctionDef: pull its source segment and build a
     per-function AST graph via src/parser/ast_graph_builder.build_ast_graph
     (same pre-order DFS used for training/inference)
  4. Run the model (loaded once) over each graph, mean-pool node logits -> sigmoid
  5. Bucket at RISK_THRESHOLD, same as the live API

Usage:
    python -m src.scan.repo_scan <path-to-repo> [top_n]
"""

import ast
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_ROOT / "src" / "parser"))

from gnn import CodeRiskGNN                          # noqa: E402
from ast_graph_builder import build_ast_graph, VOCAB_SIZE  # noqa: E402
from parse import parse_file                          # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
# Same model + threshold as src/api/app.py, so scan scores line up with /predict.
MODEL_PATH     = _ROOT / "data" / "model.pt"
HIDDEN_DIM     = 64
RISK_THRESHOLD = 0.5417

SKIP_DIRS = {"venv", "node_modules", ".git", "__pycache__"}


# ── Filesystem walk ──────────────────────────────────────────────────────────

def _iter_python_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in files:
            if filename.endswith(".py"):
                yield os.path.join(root, filename)


# ── Function extraction (reuses src/parser/parse.py for parsing) ────────────

def _extract_functions(filepath: str):
    """
    Return [(name, lineno, source_segment), ...] for every function/async
    function defined in filepath, or None if the file fails to parse.
    """
    try:
        tree = parse_file(filepath)
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except (SyntaxError, OSError, ValueError, UnicodeDecodeError):
        return None

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source, node)
            if segment is None:
                continue
            functions.append((node.name, node.lineno, segment))
    return functions


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_path=MODEL_PATH) -> CodeRiskGNN:
    device = torch.device("cpu")
    model = CodeRiskGNN(VOCAB_SIZE, HIDDEN_DIM, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def _score_function(model: CodeRiskGNN, code: str):
    """Return a risk score in [0, 1], or None if the code can't be graphed."""
    data = build_ast_graph(code)
    if data is None:
        return None
    with torch.no_grad():
        node_logits = model(data.x, data.edge_index)  # [N, 1]
        score = torch.sigmoid(node_logits.mean()).item()
    return score


# ── Repo scan ─────────────────────────────────────────────────────────────────

def scan_repo(repo_path: str, model: CodeRiskGNN = None,
               threshold: float = RISK_THRESHOLD, top_n: int = 10) -> dict:
    """
    Walk repo_path, score every function, and return:
      {
        "functions": [{file_path, function_name, lineno, risk_score, label_at_threshold}, ...],
        "summary": {
            "total_functions", "flagged_count", "threshold",
            "files_scanned", "file_parse_failures", "function_graph_failures",
            "top_riskiest": [...]
        }
      }
    """
    repo_path = str(repo_path)
    if model is None:
        model = load_model()

    functions_out = []
    files_scanned = 0
    file_parse_failures = 0
    function_graph_failures = 0

    for filepath in _iter_python_files(repo_path):
        functions = _extract_functions(filepath)
        if functions is None:
            file_parse_failures += 1
            continue
        files_scanned += 1

        rel_path = os.path.relpath(filepath, repo_path).replace(os.sep, "/")
        for name, lineno, code in functions:
            score = _score_function(model, code)
            if score is None:
                function_graph_failures += 1
                continue
            functions_out.append({
                "file_path": rel_path,
                "function_name": name,
                "lineno": lineno,
                "risk_score": round(score, 4),
                "label_at_threshold": int(score >= threshold),
            })

    flagged_count = sum(f["label_at_threshold"] for f in functions_out)
    top_riskiest = sorted(functions_out, key=lambda f: f["risk_score"], reverse=True)[:top_n]

    summary = {
        "total_functions": len(functions_out),
        "flagged_count": flagged_count,
        "threshold": threshold,
        "files_scanned": files_scanned,
        "file_parse_failures": file_parse_failures,
        "function_graph_failures": function_graph_failures,
        "top_riskiest": top_riskiest,
    }

    return {"functions": functions_out, "summary": summary}


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model()

    print(f"Scanning {target} ...")
    report = scan_repo(target, model=model, top_n=top_n)

    print(json.dumps(report["summary"], indent=2))
