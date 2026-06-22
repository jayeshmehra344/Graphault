"""
app.py — Graphault FastAPI service
==================================
Three endpoints:
  POST /predict      -> risk score 0..1 for a pasted function
  POST /explain      -> risk score + which AST nodes drove it (the crown jewel)
  GET  /model-info   -> model metadata for the dashboard / interviewers

Design notes:
- Model + builder are loaded ONCE at startup (not per request).
- Everything runs on CPU here — inference on one small graph is trivial, no GPU
  needed for serving. Your RTX 3050 Ti is for training only.
- Written model-AGNOSTIC: when you swap 89-dim one-hot -> 768-dim CodeBERT,
  you change ONLY build_graph + the model's first layer. This file does NOT change.

Run locally:
    pip install fastapi uvicorn torch torch-geometric pydantic
    uvicorn src.api.app:app --reload --port 8000
Then open http://localhost:8000/docs for the auto Swagger UI.
"""

import sys
import ast
import torch
import torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path wiring ─────────────────────────────────────────────────────────────
# Makes src/model and src/parser importable regardless of working directory.
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_ROOT / "src" / "parser"))
sys.path.insert(0, str(_ROOT / "src" / "scan"))

# ── Placeholder 1: real model import ────────────────────────────────────────
from gnn import CodeRiskGNN                         # nn.Module, returns raw logits [N, 1]

# ── Placeholder 2: real graph builder import ────────────────────────────────
from ast_graph_builder import build_ast_graph       # code:str -> PyG Data | None

# ── Repo scanner: walks a local folder, scores every function in-process ────
from repo_scan import scan_repo                     # (repo_path, model, threshold) -> {"functions": [...], "summary": {...}}

# ----------------------------------------------------------------------
MODEL_PATH = str(_ROOT / "data" / "model.pt")
NODE_FEATURE_DIM = 89          # change to 768 when CodeBERT lands — only place it matters
HIDDEN_DIM = 64                # match whatever you trained with

# ----------------------------------------------------------------------
# Load model once at startup
# ----------------------------------------------------------------------
device = torch.device("cpu")
model = None

def load_model():
    # ── Placeholder 1 wired ─────────────────────────────────────────────
    m = CodeRiskGNN(NODE_FEATURE_DIM, HIDDEN_DIM, 1)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.to(device).eval()
    return m


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Graphault", description="GNN code vulnerability risk predictor", version="0.1")

# CORS so your React dashboard (different port/origin) can call this.
# Lock allow_origins down to your real frontend URL before AWS deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://graphault-frontend.s3-website.ap-south-1.amazonaws.com",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    global model
    model = load_model()


# ---- request/response schemas ----
class CodeRequest(BaseModel):
    code: str

class PredictResponse(BaseModel):
    risk_score: float
    label: int          # 0 clean / 1 risky, at threshold
    num_nodes: int

class NodeContribution(BaseModel):
    node_index: int
    node_type: str      # e.g. "Call", "If", "Compare"
    lineno: int | None
    contribution: float # relative importance, 0..1

class ExplainResponse(BaseModel):
    risk_score: float
    label: int
    top_nodes: list[NodeContribution]


class ScanRequest(BaseModel):
    repo_path: str

class FunctionRisk(BaseModel):
    file_path: str
    function_name: str
    lineno: int
    risk_score: float
    label_at_threshold: int

class ScanSummary(BaseModel):
    total_functions: int
    flagged_count: int
    threshold: float
    files_scanned: int
    file_parse_failures: int
    function_graph_failures: int
    top_riskiest: list[FunctionRisk]

class ScanResponse(BaseModel):
    summary: ScanSummary
    functions: list[FunctionRisk]


# threshold: tune to your validation PR curve, don't leave at 0.5 for 12:1 imbalance
RISK_THRESHOLD = 0.5417  # tuned to maximise val-F1 via find_threshold.py


def _code_to_data(code: str):
    """Parse + build the PyG graph. Raises 422 on unparseable code."""
    # ── Placeholder 2 wired ─────────────────────────────────────────────
    # build_ast_graph returns None on SyntaxError / empty code
    try:
        data = build_ast_graph(code)
    except Exception:
        data = None
    if data is None:
        raise HTTPException(status_code=422, detail="Could not parse: invalid Python syntax.")
    return data


@app.post("/predict", response_model=PredictResponse)
def predict(req: CodeRequest):
    data = _code_to_data(req.code)
    with torch.no_grad():
        # ── Placeholder 3 wired ─────────────────────────────────────────
        # model returns per-node logits [N, 1]; mean-pool to scalar for graph classification
        node_logits = model(data.x.to(device), data.edge_index.to(device))  # [N, 1]
        logit = node_logits.mean()
        score = torch.sigmoid(logit).item()
    return PredictResponse(
        risk_score=round(score, 4),
        label=int(score >= RISK_THRESHOLD),
        num_nodes=data.x.size(0),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: CodeRequest):
    """
    The crown jewel: WHY did the model flag this function.
    Uses a gradient-based saliency over node features — magnitude of d(score)/d(x)
    per node = how much that node pushed the prediction. Simple, fast, no extra deps.
    For a fancier version later, swap in PyG's GNNExplainer (same endpoint, no API change).
    """
    data = _code_to_data(req.code)

    x = data.x.clone().to(device).requires_grad_(True)
    # ── Placeholder 3 wired (explain path) ──────────────────────────────
    node_logits = model(x, data.edge_index.to(device))  # [N, 1]
    logit = node_logits.mean()
    score = torch.sigmoid(logit)
    score.backward()

    # per-node importance = L2 norm of that node's feature gradient
    node_saliency = x.grad.norm(dim=1)             # [num_nodes]
    if node_saliency.max() > 0:
        node_saliency = node_saliency / node_saliency.max()   # normalize 0..1

    # map node index -> ast type + line (same walk order as builder)
    nodes = list(ast.walk(ast.parse(req.code)))
    contribs = []
    for i, sal in enumerate(node_saliency.tolist()):
        n = nodes[i] if i < len(nodes) else None
        contribs.append(NodeContribution(
            node_index=i,
            node_type=type(n).__name__ if n else "?",
            lineno=getattr(n, "lineno", None) if n else None,
            contribution=round(sal, 4),
        ))
    contribs.sort(key=lambda c: c.contribution, reverse=True)

    return ExplainResponse(
        risk_score=round(score.item(), 4),
        label=int(score.item() >= RISK_THRESHOLD),
        top_nodes=contribs[:10],          # top 10 most influential nodes
    )


@app.post("/scan-repo", response_model=ScanResponse)
def scan_repo_endpoint(req: ScanRequest):
    """
    Scan a local Python repo/folder: extract every function, build its AST
    graph, and score it with the already-loaded model (no reload, no MongoDB).
    Delegates the actual walk + scoring to src/scan/repo_scan.scan_repo.
    """
    repo_path = Path(req.repo_path)
    if not repo_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.repo_path}")
    if not repo_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.repo_path}")

    try:
        report = scan_repo(str(repo_path), model=model, threshold=RISK_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")

    return ScanResponse(**report)


@app.get("/model-info")
def model_info():
    """Metadata for the dashboard + something concrete to show interviewers."""
    return {
        "model": "Graphault GNN (per-function AST / Code Property Graph)",
        "task": "graph classification (function -> vulnerability risk 0/1)",
        "approach": "Devign/ReVeal-style per-function CPG",
        "node_features": f"{NODE_FEATURE_DIM}-dim AST node-type one-hot (CodeBERT upgrade planned)",
        "val_pr_auc": 0.2414,
        "val_f1": 0.27,
        "random_baseline_pr_auc": 0.075,
        "uplift_over_random": "~3.2x",
        "train_val_name_overlap": 0,
        "class_imbalance_pos_weight": 12.32,
        "threshold": RISK_THRESHOLD,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
