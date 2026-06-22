"""
Precompute CodeBERT node features for all functions in train+test splits.

Output: data/codebert_node_features.pt
  {str(_id): FloatTensor[num_nodes, 768]}

Node order matches the pre-order DFS from _assign_node_ids.
"""

import ast
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

import torch
from bson import ObjectId
from transformers import AutoTokenizer, AutoModel

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_ROOT / "src" / "graph"))

from db import get_db

# ── Inlined from ast_graph_builder (avoids torch_geometric dependency) ─────────

NODE_TYPES = [
    'Module', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef',
    'Return', 'Delete', 'Assign', 'AugAssign', 'AnnAssign',
    'For', 'AsyncFor', 'While', 'If', 'With', 'AsyncWith',
    'Raise', 'Try', 'ExceptHandler', 'Assert',
    'Import', 'ImportFrom', 'Global', 'Nonlocal',
    'Expr', 'Pass', 'Break', 'Continue',
    'BoolOp', 'NamedExpr', 'BinOp', 'UnaryOp', 'Lambda', 'IfExp',
    'Dict', 'Set', 'List', 'Tuple',
    'ListComp', 'SetComp', 'DictComp', 'GeneratorExp',
    'Await', 'Yield', 'YieldFrom',
    'Compare', 'Call', 'FormattedValue', 'JoinedStr',
    'Constant', 'Attribute', 'Subscript', 'Starred', 'Name', 'Slice',
    'And', 'Or',
    'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn',
    'Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow', 'FloorDiv', 'MatMult',
    'BitOr', 'BitAnd', 'BitXor', 'LShift', 'RShift',
    'Not', 'Invert', 'UAdd', 'USub',
    'arguments', 'arg', 'keyword',
    'alias', 'withitem',
    'UNKNOWN',
]
NODE_TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(NODE_TYPES)}


def _find_func_root(tree: ast.AST) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    if isinstance(tree, ast.Module) and tree.body:
        return tree
    return None


def _assign_node_ids(root: ast.AST) -> tuple[dict, list[int]]:
    node_ids: dict[int, int] = {}
    type_indices: list[int] = []

    def _visit(node: ast.AST):
        key = id(node)
        if key in node_ids:
            return
        node_ids[key] = len(node_ids)
        type_indices.append(NODE_TYPE_INDEX.get(type(node).__name__, NODE_TYPE_INDEX['UNKNOWN']))
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(root)
    return node_ids, type_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_ROOT / "data" / "precompute_codebert.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

CODEBERT_MODEL = "microsoft/codebert-base"
CACHE_PATH = Path("D:/graphault_cache/codebert_node_features.pt")
SPLITS_DIR = _ROOT / "data" / "splits"
DIM = 768
FETCH_BATCH = 5_000
SAVE_EVERY = 500


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_code_with_source(code: str) -> tuple[Optional[ast.AST], Optional[str]]:
    """Returns (tree, source_used) — source_used is the string that was parsed."""
    code = textwrap.dedent(code)
    for attempt in (code, f"def _wrapper():\n{textwrap.indent(code, '    ')}"):
        try:
            return ast.parse(attempt), attempt
        except SyntaxError:
            continue
    return None, None


def _char_span(node: ast.AST, line_starts: list[int]) -> Optional[tuple[int, int]]:
    """Returns (start_char, end_char) for an AST node, or None if no position info."""
    try:
        start = line_starts[node.lineno - 1] + node.col_offset
        end   = line_starts[node.end_lineno - 1] + node.end_col_offset
        return (start, end)
    except (AttributeError, IndexError):
        return None


# ── Feature computation ────────────────────────────────────────────────────────

def compute_features(
    code: str,
    tokenizer,
    model,
    structural_vec: torch.Tensor,
    device: torch.device,
) -> Optional[tuple[torch.Tensor, list[int]]]:
    """
    Returns (FloatTensor[num_nodes, 768], type_indices) or None if parsing fails.
    Node order is pre-order DFS matching _assign_node_ids.
    """
    tree, source = _parse_code_with_source(code)
    if tree is None:
        return None

    root = _find_func_root(tree)
    if root is None:
        return None

    node_ids, type_indices = _assign_node_ids(root)
    num_nodes = len(node_ids)
    if num_nodes == 0:
        return None

    # Precompute cumulative line start offsets for char-span lookup
    lines = source.splitlines(keepends=True)
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    # Single AST walk: build span map and children map simultaneously
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

    # Tokenize the same source string that was parsed (so char offsets align)
    enc = tokenizer(
        source,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        return_offsets_mapping=True,
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()  # [(start_char, end_char)]

    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        token_emb = model(**enc).last_hidden_state[0].cpu()  # [seq_len, 768] on CPU

    features = torch.zeros(num_nodes, DIM)
    has_feature = [False] * num_nodes

    # Vectorized token→node overlap via tensor broadcasting.
    # Collect positioned nodes (those with valid char spans).
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
        # Real tokens only (start < end filters out special tokens like CLS/SEP)
        real_idx = [i for i, (s, e) in enumerate(offset_mapping) if s < e]
        if real_idx:
            t_starts = torch.tensor([offset_mapping[i][0] for i in real_idx])  # [T]
            t_ends   = torch.tensor([offset_mapping[i][1] for i in real_idx])  # [T]
            t_emb    = token_emb[real_idx]                                       # [T, 768]

            n_starts = torch.tensor(p_starts_l)  # [P]
            n_ends   = torch.tensor(p_ends_l)    # [P]

            # overlap[i, j] = node i overlaps token j  →  [P, T] bool
            overlap = (n_starts[:, None] < t_ends[None, :]) & \
                      (t_starts[None, :] < n_ends[:, None])

            counts = overlap.float().sum(dim=1)          # [P]
            has_tok = counts > 0

            if has_tok.any():
                # [P, T] @ [T, 768] → [P, 768]; divide by count for mean
                agg = overlap.float() @ t_emb            # [P, 768]
                agg[has_tok] = agg[has_tok] / counts[has_tok, None]

                for i, nid in enumerate(p_nids):
                    if has_tok[i].item():
                        features[nid] = agg[i]
                        has_feature[nid] = True

    # Bottom-up fill for nodes with no overlapping tokens.
    # Pre-order DFS guarantees parent_index < all_descendant_indices,
    # so reversed order processes children before their parents.
    for nid in reversed(range(num_nodes)):
        if has_feature[nid]:
            continue
        child_feats = [features[c] for c in children[nid] if has_feature[c]]
        if child_feats:
            features[nid] = torch.stack(child_feats).mean(dim=0)
            has_feature[nid] = True
        else:
            features[nid] = structural_vec

    return features, type_indices


# ── Save helper ───────────────────────────────────────────────────────────────

def _safe_save(cache: dict, path: Path) -> bool:
    """Write to a temp file then rename — prevents corruption on partial write."""
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(cache, tmp)
        tmp.replace(path)
        return True
    except Exception as e:
        log.warning(f"Checkpoint save failed: {e}")
        tmp.unlink(missing_ok=True)
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Collect all IDs from both splits (deduplicated, order preserved)
    all_ids: list[str] = []
    seen: set[str] = set()
    for split_file in ("train_ids.json", "test_ids.json"):
        with open(SPLITS_DIR / split_file) as f:
            for sid in json.load(f):
                if sid not in seen:
                    all_ids.append(sid)
                    seen.add(sid)

    log.info(f"Total unique IDs across splits: {len(all_ids):,}")

    # Resume from existing cache
    cache: dict[str, torch.Tensor] = {}
    if CACHE_PATH.exists():
        log.info(f"Loading existing cache from {CACHE_PATH}")
        try:
            cache = torch.load(CACHE_PATH, weights_only=True)
        except TypeError:
            cache = torch.load(CACHE_PATH)
        log.info(f"  Resuming: {len(cache):,} already computed")

    remaining = [sid for sid in all_ids if sid not in cache]
    log.info(f"Remaining: {len(remaining):,}")

    if not remaining:
        log.info("All functions already computed. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Load CodeBERT
    log.info(f"Loading {CODEBERT_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model = AutoModel.from_pretrained(CODEBERT_MODEL).to(device)
    model.eval()

    # Fixed structural vector: mean of word embedding weights, kept on CPU
    structural_vec = model.embeddings.word_embeddings.weight.mean(dim=0).detach().cpu()
    log.info(f"Structural vec norm: {structural_vec.norm():.4f}")

    db = get_db()
    collection = db["labeled_functions"]
    object_ids = [ObjectId(sid) for sid in remaining]

    processed = 0
    skipped = 0
    verification_done = False

    for batch_start in range(0, len(object_ids), FETCH_BATCH):
        batch_oids = object_ids[batch_start : batch_start + FETCH_BATCH]
        batch_sids = remaining[batch_start : batch_start + FETCH_BATCH]

        docs_by_id = {
            str(d["_id"]): d
            for d in collection.find(
                {"_id": {"$in": batch_oids}},
                {"code": 1, "_id": 1},
            )
        }

        for sid in batch_sids:
            doc = docs_by_id.get(sid)
            if doc is None:
                skipped += 1
                continue

            result = compute_features(doc.get("code", ""), tokenizer, model, structural_vec, device)
            if result is None:
                skipped += 1
                continue

            feats, type_indices = result
            cache[sid] = feats
            processed += 1

            # Verify first 5: log DFS node types to confirm pre-order traversal
            if not verification_done and processed <= 5:
                dfs_types = [NODE_TYPES[t] for t in type_indices[:8]]
                log.info(
                    f"  [verify {processed}] id={sid}  "
                    f"nodes={feats.shape[0]}  "
                    f"mean_norm={feats.norm(dim=1).mean():.4f}  "
                    f"dfs_prefix={dfs_types}"
                )
                if processed == 5:
                    verification_done = True

            if processed % SAVE_EVERY == 0:
                _safe_save(cache, CACHE_PATH)
                pct = 100.0 * processed / len(remaining)
                log.info(
                    f"  [{processed:,}/{len(remaining):,} {pct:.1f}%] "
                    f"skipped={skipped}  cache_size={len(cache):,}"
                )

    torch.save(cache, CACHE_PATH)
    log.info(
        f"Finished. processed={processed:,}  skipped={skipped}  "
        f"total_in_cache={len(cache):,}  saved={CACHE_PATH}"
    )


if __name__ == "__main__":
    main()
