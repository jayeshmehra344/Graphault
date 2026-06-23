"""
app.py — Graphault FastAPI service
==================================
Endpoints:
  POST /predict      -> risk score 0..1 (CodeBERT by default; ?model=89dim for one-hot)
  POST /explain      -> risk score + which AST nodes drove it (89-dim model only)
  POST /scan-repo    -> whole-repo scan (89-dim model; CodeBERT per-function would be too slow)
  GET  /model-info   -> model metadata

Query param ?model=codebert (default) | ?model=89dim selects the model for /predict and /model-info.

LOCAL ONLY NOTE: CodeBERT serving (~500 MB encoder on CPU) fits on a 16 GB laptop but will OOM
on a t3.micro EC2 (1 GB RAM). The 89-dim path is what the EC2 should run.

Run locally:
    uvicorn src.api.app:app --reload --port 8000
"""

import ast
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path wiring ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_ROOT / "src" / "parser"))
sys.path.insert(0, str(_ROOT / "src" / "scan"))

from gnn import CodeRiskGNN
from ast_graph_builder import (
    build_ast_graph,
    build_graph_skeleton,
    get_dfs_ordered_nodes,
    NODE_TYPE_INDEX,
    VOCAB_SIZE,
)
from repo_scan import scan_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Model paths ──────────────────────────────────────────────────────────────
MODEL_89DIM_PATH    = _ROOT / "data" / "model_deduped_89dim.pt"
MODEL_CODEBERT_PATH = _ROOT / "data" / "model_deduped_codebert.pt"

HIDDEN_DIM          = 64
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODEBERT_DIM        = 768

RISK_THRESHOLD_89DIM    = 0.5417   # F1-optimal for 89-dim deduped model
RISK_THRESHOLD_CODEBERT = 0.683    # F1-optimal for CodeBERT model

device = torch.device("cpu")

# ── Global model handles ─────────────────────────────────────────────────────
model_89dim         = None   # CodeRiskGNN(89, 64, 1)
model_codebert_gnn  = None   # CodeRiskGNN(768, 64, 1)
codebert_tokenizer  = None   # AutoTokenizer
codebert_encoder    = None   # AutoModel (microsoft/codebert-base)
codebert_struct_vec = None   # mean word-embedding vector, CPU; fallback for spanless nodes


# ── CodeBERT feature helpers ─────────────────────────────────────────────────
# Logic mirrors precompute_codebert.py exactly so the DFS node order and
# bottom-up fill match what was used at training time.

def _parse_with_source(code: str) -> tuple[Optional[ast.AST], Optional[str]]:
    """Parse code; return (tree, source_actually_parsed) or (None, None)."""
    code = textwrap.dedent(code)
    for attempt in (code, f"def _wrapper():\n{textwrap.indent(code, '    ')}"):
        try:
            return ast.parse(attempt), attempt
        except SyntaxError:
            continue
    return None, None


def _find_func_root(tree: ast.AST) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    if isinstance(tree, ast.Module) and tree.body:
        return tree
    return None


def _dfs_node_ids(root: ast.AST) -> tuple[dict, list[int]]:
    """Pre-order DFS — same as _assign_node_ids in precompute_codebert.py."""
    node_ids: dict[int, int] = {}
    type_indices: list[int] = []

    def _visit(node: ast.AST):
        key = id(node)
        if key in node_ids:
            return
        node_ids[key] = len(node_ids)
        type_indices.append(NODE_TYPE_INDEX.get(type(node).__name__, NODE_TYPE_INDEX["UNKNOWN"]))
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(root)
    return node_ids, type_indices


def _char_span(node: ast.AST, line_starts: list) -> Optional[tuple[int, int]]:
    try:
        start = line_starts[node.lineno - 1] + node.col_offset
        end   = line_starts[node.end_lineno - 1] + node.end_col_offset
        return (start, end)
    except (AttributeError, IndexError):
        return None


def _compute_codebert_features(code: str) -> Optional[torch.Tensor]:
    """
    Compute 768-dim CodeBERT node features for one code snippet.

    Returns FloatTensor[num_nodes, 768], L2-normalised per node (matching
    CodeBERTFP16MmapDataset.__getitem__ in train.py).
    Node order is pre-order DFS from _dfs_node_ids, identical to build_graph_skeleton.
    Returns None if parsing fails or CodeBERT encoder isn't loaded.
    """
    if codebert_tokenizer is None or codebert_encoder is None:
        return None

    tree, source = _parse_with_source(code)
    if tree is None:
        return None

    root = _find_func_root(tree)
    if root is None:
        return None

    node_ids, _ = _dfs_node_ids(root)
    num_nodes = len(node_ids)
    if num_nodes == 0:
        return None

    # Build line-start offsets so AST char spans align with tokenizer offsets.
    lines = source.splitlines(keepends=True)
    line_starts: list[int] = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    # Single AST walk: collect char spans and parent→child relationships.
    nid_to_span: dict[int, Optional[tuple[int, int]]] = {}
    children: dict[int, list[int]] = {i: [] for i in range(num_nodes)}

    for node in ast.walk(root):
        nid = node_ids.get(id(node))
        if nid is None:
            continue
        nid_to_span[nid] = _char_span(node, line_starts)
        for child in ast.iter_child_nodes(node):
            cid = node_ids.get(id(child))
            if cid is not None:
                children[nid].append(cid)

    # Tokenize the parsed source (same string whose char spans we computed above).
    enc = codebert_tokenizer(
        source,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        return_offsets_mapping=True,
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        token_emb = codebert_encoder(**enc).last_hidden_state[0].cpu()  # [seq_len, 768]

    features    = torch.zeros(num_nodes, CODEBERT_DIM)
    has_feature = [False] * num_nodes

    # Collect nodes that have a valid char span (positioned nodes).
    p_nids, p_starts_l, p_ends_l = [], [], []
    for nid in range(num_nodes):
        span = nid_to_span.get(nid)
        if span is not None:
            ns, ne = span
            if ns < ne:
                p_nids.append(nid)
                p_starts_l.append(ns)
                p_ends_l.append(ne)

    if p_nids:
        # Filter to real (non-special) tokens.
        real_idx = [i for i, (s, e) in enumerate(offset_mapping) if s < e]
        if real_idx:
            t_starts = torch.tensor([offset_mapping[i][0] for i in real_idx])  # [T]
            t_ends   = torch.tensor([offset_mapping[i][1] for i in real_idx])  # [T]
            t_emb    = token_emb[real_idx]                                       # [T, 768]

            n_starts = torch.tensor(p_starts_l)  # [P]
            n_ends   = torch.tensor(p_ends_l)    # [P]

            # overlap[i, j] = True if node i overlaps with token j.
            overlap = (n_starts[:, None] < t_ends[None, :]) & \
                      (t_starts[None, :] < n_ends[:, None])

            counts  = overlap.float().sum(dim=1)   # [P]
            has_tok = counts > 0

            if has_tok.any():
                agg = overlap.float() @ t_emb           # [P, 768] sum
                agg[has_tok] = agg[has_tok] / counts[has_tok, None]  # mean

                for i, nid in enumerate(p_nids):
                    if has_tok[i].item():
                        features[nid] = agg[i]
                        has_feature[nid] = True

    # Bottom-up fill: nodes with no overlapping tokens average their children's features.
    # Reversed pre-order DFS = children before parents.
    for nid in reversed(range(num_nodes)):
        if has_feature[nid]:
            continue
        child_feats = [features[c] for c in children[nid] if has_feature[c]]
        if child_feats:
            features[nid] = torch.stack(child_feats).mean(dim=0)
            has_feature[nid] = True
        else:
            features[nid] = codebert_struct_vec   # structural fallback

    # L2 normalise per node — matches CodeBERTFP16MmapDataset.__getitem__ in train.py.
    return F.normalize(features, p=2, dim=1)


def _code_to_codebert_data(code: str):
    """Build a PyG Data with 768-dim CodeBERT features. Raises HTTPException on failure."""
    from torch_geometric.data import Data

    feats = _compute_codebert_features(code)
    if feats is None:
        if codebert_tokenizer is None:
            raise HTTPException(
                status_code=503,
                detail="CodeBERT encoder not loaded. Try ?model=89dim or restart the server.",
            )
        raise HTTPException(status_code=422, detail="Could not parse: invalid Python syntax.")

    skeleton = build_graph_skeleton(code)
    if skeleton is None:
        raise HTTPException(status_code=422, detail="Could not parse: invalid Python syntax.")

    edge_index, edge_attr, y, n_nodes = skeleton
    if feats.shape[0] != n_nodes:
        raise HTTPException(
            status_code=500,
            detail=f"Node count mismatch: CodeBERT={feats.shape[0]}, AST={n_nodes}. "
                   "This indicates a DFS-order bug — please report.",
        )

    return Data(x=feats, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ── Load models at startup ───────────────────────────────────────────────────

def _load_models():
    global model_89dim, model_codebert_gnn, codebert_tokenizer, codebert_encoder, codebert_struct_vec

    # 89-dim model (fast, no external deps).
    try:
        m = CodeRiskGNN(VOCAB_SIZE, HIDDEN_DIM, 1)
        m.load_state_dict(torch.load(MODEL_89DIM_PATH, map_location=device))
        model_89dim = m.to(device).eval()
        log.info(f"89-dim model loaded from {MODEL_89DIM_PATH.name}")
    except Exception as exc:
        log.error(f"89-dim model failed to load: {exc}")

    # CodeBERT GNN weights (small checkpoint — fast).
    try:
        mcb = CodeRiskGNN(CODEBERT_DIM, HIDDEN_DIM, 1)
        mcb.load_state_dict(torch.load(MODEL_CODEBERT_PATH, map_location=device))
        model_codebert_gnn = mcb.to(device).eval()
        log.info(f"CodeBERT GNN weights loaded from {MODEL_CODEBERT_PATH.name}")
    except Exception as exc:
        log.warning(f"CodeBERT GNN weights failed to load (CodeBERT unavailable): {exc}")

    # CodeBERT tokenizer + encoder (~500 MB; may take a few seconds from HF cache).
    # DOES NOT fit on t3.micro — local serving only.
    try:
        from transformers import AutoTokenizer, AutoModel
        log.info(f"Loading {CODEBERT_MODEL_NAME} tokenizer + encoder (may take a few seconds)…")
        codebert_tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        enc = AutoModel.from_pretrained(CODEBERT_MODEL_NAME).to(device)
        enc.eval()
        codebert_encoder   = enc
        # Structural fallback vector: mean of word embedding weights.
        codebert_struct_vec = enc.embeddings.word_embeddings.weight.mean(dim=0).detach().cpu()
        log.info("CodeBERT tokenizer + encoder ready")
    except Exception as exc:
        log.warning(
            f"CodeBERT encoder failed to load — CodeBERT predictions unavailable. "
            f"Use ?model=89dim as a fallback. Error: {exc}"
        )


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Graphault",
    description="GNN code vulnerability risk predictor (CodeBERT or 89-dim one-hot features)",
    version="0.2",
)

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
    _load_models()


# ── Request / response schemas ───────────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str

class PredictResponse(BaseModel):
    risk_score: float
    label: int
    num_nodes: int
    model_used: str

class NodeContribution(BaseModel):
    node_index: int
    node_type: str
    lineno: Optional[int]
    contribution: float

class ExplainResponse(BaseModel):
    risk_score: float
    label: int
    top_nodes: list[NodeContribution]
    model_used: str

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


# ── Shared helper: 89-dim graph build ───────────────────────────────────────

def _code_to_data_89(code: str):
    """Build a 89-dim one-hot PyG Data object. Raises 422 on unparseable code."""
    try:
        data = build_ast_graph(code)
    except Exception:
        data = None
    if data is None:
        raise HTTPException(status_code=422, detail="Could not parse: invalid Python syntax.")
    return data


# ── /predict ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: CodeRequest,
    model: str = Query(
        "codebert",
        description="Feature model to use: 'codebert' (768-dim, local only) or '89dim' (one-hot)",
    ),
):
    """
    Risk-score one Python function.
    Default is CodeBERT (768-dim) which is what the dashboard expects.
    Pass ?model=89dim if CodeBERT isn't loaded (EC2, slow machine, etc.).
    """
    if model == "codebert":
        if model_codebert_gnn is None or codebert_tokenizer is None:
            raise HTTPException(
                status_code=503,
                detail="CodeBERT model not loaded. Use ?model=89dim or restart the server.",
            )
        data      = _code_to_codebert_data(req.code)
        gnn       = model_codebert_gnn
        threshold = RISK_THRESHOLD_CODEBERT
        label_str = "codebert"
    else:
        if model_89dim is None:
            raise HTTPException(status_code=503, detail="89-dim model not loaded.")
        data      = _code_to_data_89(req.code)
        gnn       = model_89dim
        threshold = RISK_THRESHOLD_89DIM
        label_str = "89dim"

    with torch.no_grad():
        node_logits = gnn(data.x.to(device), data.edge_index.to(device))  # [N, 1]
        score = torch.sigmoid(node_logits.mean()).item()

    return PredictResponse(
        risk_score=round(score, 4),
        label=int(score >= threshold),
        num_nodes=data.x.size(0),
        model_used=label_str,
    )


# ── /explain ─────────────────────────────────────────────────────────────────

@app.post("/explain", response_model=ExplainResponse)
def explain(req: CodeRequest):
    """
    Risk score + gradient saliency over AST nodes.

    Uses CodeBERT GNN when the encoder is loaded (the normal local case).
    CodeBERT features are computed forward-only (no_grad), then the resulting
    feature matrix is cloned with requires_grad=True so the GNN backward pass
    gives per-node saliency. This does NOT backprop through the transformer —
    only through the 3-layer GNN — so it is fast on CPU.
    Falls back to 89-dim if CodeBERT encoder is not loaded.
    """
    use_codebert = model_codebert_gnn is not None and codebert_tokenizer is not None

    if use_codebert:
        data      = _code_to_codebert_data(req.code)
        gnn       = model_codebert_gnn
        threshold = RISK_THRESHOLD_CODEBERT
        model_label = "codebert"
    else:
        if model_89dim is None:
            raise HTTPException(status_code=503, detail="No model loaded.")
        data      = _code_to_data_89(req.code)
        gnn       = model_89dim
        threshold = RISK_THRESHOLD_89DIM
        model_label = "89dim"

    x = data.x.clone().to(device).requires_grad_(True)
    node_logits = gnn(x, data.edge_index.to(device))   # [N, 1]
    score = torch.sigmoid(node_logits.mean())
    score.backward()

    node_saliency = x.grad.norm(dim=1)              # [N]
    if node_saliency.max() > 0:
        node_saliency = node_saliency / node_saliency.max()

    nodes = get_dfs_ordered_nodes(req.code)  # pre-order DFS from func root, matching graph node indices
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
        label=int(score.item() >= threshold),
        top_nodes=contribs[:10],
        model_used=model_label,
    )


# ── /scan-repo ────────────────────────────────────────────────────────────────

@app.post("/scan-repo", response_model=ScanResponse)
def scan_repo_endpoint(req: ScanRequest):
    """
    Scan a local Python repo/folder.
    Uses the 89-dim model — on-the-fly CodeBERT inference for every function
    in a repo would be O(minutes) on CPU.
    """
    repo_path = Path(req.repo_path)
    if not repo_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.repo_path}")
    if not repo_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.repo_path}")
    if model_89dim is None:
        raise HTTPException(status_code=503, detail="89-dim model not loaded.")

    try:
        report = scan_repo(str(repo_path), model=model_89dim, threshold=RISK_THRESHOLD_89DIM)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scan failed: {exc}")

    return ScanResponse(**report)


# ── /model-info ───────────────────────────────────────────────────────────────

@app.get("/model-info")
def model_info(
    model: str = Query(
        "codebert",
        description="'codebert' or '89dim'",
    ),
):
    """Report metadata for whichever model is selected."""
    if model == "codebert":
        return {
            "model": "Graphault GNN + CodeBERT (768-dim node embeddings)",
            "task": "graph classification — function → vulnerability risk 0/1",
            "approach": "Devign/ReVeal-style per-function CPG; node features from microsoft/codebert-base",
            "node_features": (
                "768-dim CodeBERT embeddings computed on-the-fly: "
                "tokenise → mean-overlap per AST node → bottom-up fill → L2-norm"
            ),
            "val_pr_auc": 0.2318,
            "threshold": RISK_THRESHOLD_CODEBERT,
            "random_baseline_pr_auc": "~0.075",
            "uplift_over_random": "~3.1x",
            "serving_note": (
                "LOCAL ONLY — CodeBERT encoder (~500 MB) will OOM on t3.micro EC2 (1 GB RAM). "
                "Use ?model=89dim on EC2."
            ),
            "codebert_loaded": model_codebert_gnn is not None and codebert_tokenizer is not None,
            "model_file": MODEL_CODEBERT_PATH.name,
        }
    else:
        return {
            "model": "Graphault GNN (89-dim one-hot AST node type)",
            "task": "graph classification — function → vulnerability risk 0/1",
            "approach": "Devign/ReVeal-style per-function CPG",
            "node_features": f"{VOCAB_SIZE}-dim AST node-type one-hot",
            "val_pr_auc": 0.2414,
            "val_f1": 0.27,
            "threshold": RISK_THRESHOLD_89DIM,
            "random_baseline_pr_auc": "~0.075",
            "uplift_over_random": "~3.2x",
            "serving_note": "EC2-safe — no CodeBERT encoder needed",
            "model_loaded": model_89dim is not None,
            "model_file": MODEL_89DIM_PATH.name,
        }


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_89dim_loaded":       model_89dim is not None,
        "model_codebert_loaded":    model_codebert_gnn is not None,
        "codebert_encoder_loaded":  codebert_encoder is not None,
    }
