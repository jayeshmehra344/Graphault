"""
evaluate.py — standalone eval harness for Graphault GNN checkpoints.

Reads test functions STRICTLY from test_ids.json (no re-sampling, no func_name_split).
Metric block copied verbatim from train_deduped() / train_deduped_codebert() in train.py.

Usage:
    python src/model/evaluate.py \\
        --checkpoint data/model_deduped_codebert.pt \\
        --features codebert \\
        --test-ids data/splits/test_ids.json

    python src/model/evaluate.py \\
        --checkpoint data/model_deduped_89dim.pt \\
        --features onehot \\
        --test-ids data/splits/test_ids.json

Acceptance targets:
    codebert -> PR-AUC ~0.2318, F1 ~0.282
    onehot   -> PR-AUC ~0.2042
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_ROOT / "src" / "parser"))
sys.path.insert(0, str(_ROOT / "src" / "graph"))

from gnn import CodeRiskGNN                                          # noqa: E402
from ast_graph_builder import build_ast_graph, build_graph_skeleton, VOCAB_SIZE  # noqa: E402
from db import get_db                                                # noqa: E402

# ── same paths as train.py ────────────────────────────────────────────────────
CODEBERT_DIM          = 768
HIDDEN_DIM            = 64
FP16_CACHE_DATA_PATH  = Path("D:/graphault_cache/codebert_fp16.bin")
FP16_CACHE_INDEX_PATH = Path("D:/graphault_cache/codebert_fp16_index.json")
OUTPUT_DIR            = _ROOT / "data" / "eval"


# ── verbatim from train.py:138-145 ───────────────────────────────────────────

def _find_best_threshold(labels: np.ndarray, probs: np.ndarray):
    """F1-maximising threshold from the PR curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    p, r, t = precision[:-1], recall[:-1], thresholds
    denom = p + r
    f1s = np.where(denom > 0, 2 * p * r / denom, 0.0)
    best = int(f1s.argmax())
    return float(t[best]), float(f1s[best]), float(p[best]), float(r[best])


# ── MongoDB fetch — verbatim from train.py:_fetch_docs_for_str_ids ───────────

def _fetch_docs_for_str_ids(str_ids: list) -> dict:
    """Fetch {code, label} docs from MongoDB for the given str_ids."""
    from bson import ObjectId
    object_ids = [ObjectId(s) for s in str_ids]
    db = get_db()
    collection = db["labeled_functions"]
    BATCH = 5_000
    docs_by_id = {}
    for start in range(0, len(object_ids), BATCH):
        chunk = object_ids[start:start + BATCH]
        for d in collection.find({"_id": {"$in": chunk}}, {"code": 1, "label": 1}):
            docs_by_id[str(d["_id"])] = d
        print(f"  fetched {min(start + BATCH, len(object_ids)):,}/{len(object_ids):,}", flush=True)
    return docs_by_id


# ── CodeBERT dataset — verbatim copy of class from train.py:351-390 ──────────

class CodeBERTFP16MmapDataset(torch.utils.data.Dataset):
    """
    Dataset backed by a pre-built fp16 binary + JSON index.
    numpy memmap gives OS-managed paging of the ~10 GB file.
    __getitem__ returns fp32 tensors (cast on read) — no fp16 in the model.
    Hard node-count assertion preserved: index[sid][1] == n_nodes from AST.
    """

    def __init__(self, skeletons: list, data_path: Path, index: dict):
        self.skeletons  = skeletons
        self._data_path = str(data_path)
        self._index     = index
        self._mmap      = None   # opened on first __getitem__ (lazy, stays open)

    def _get_mmap(self):
        if self._mmap is None:
            self._mmap = np.memmap(self._data_path, dtype='float16', mode='r')
        return self._mmap

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        sid, edge_index, edge_attr, y, n_nodes = self.skeletons[idx]
        el_offset, n_nodes_cached = self._index[sid]
        # Hard invariant: cache row i == AST node i (pre-order DFS order).
        assert n_nodes_cached == n_nodes, (
            f"CodeBERT feature/AST node count mismatch for {sid}: "
            f"cache has {n_nodes_cached} nodes, AST has {n_nodes} nodes"
        )
        arr = self._get_mmap()
        # .copy() converts the mmap slice to a regular numpy array before torch wraps it.
        x = torch.from_numpy(
            arr[el_offset : el_offset + n_nodes * CODEBERT_DIM]
            .reshape(n_nodes, CODEBERT_DIM)
            .copy()
        ).float()
        x = F.normalize(x, p=2, dim=1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ── dataset builders ──────────────────────────────────────────────────────────

def _build_onehot_dataset(str_ids: list, docs_by_id: dict):
    """
    Build list of 89-dim one-hot PyG Data objects for the given IDs.
    Returns (graphs, sid_list, drop_counts) where drop_counts is a dict.
    """
    graphs   = []
    sid_list = []
    missing = parse_failed = 0

    for sid in str_ids:
        doc = docs_by_id.get(sid)
        if doc is None:
            missing += 1
            continue
        g = build_ast_graph(doc.get("code", ""), label=int(doc.get("label", 0)))
        if g is None:
            parse_failed += 1
            continue
        graphs.append(g)
        sid_list.append(sid)

    drops = {"missing_db": missing, "parse_failed": parse_failed}
    return graphs, sid_list, drops


def _build_codebert_dataset(str_ids: list, docs_by_id: dict,
                             fp16_index: dict, data_path: Path):
    """
    Build CodeBERTFP16MmapDataset for the given IDs.
    IDs absent from fp16_index are counted as 'missing_fp16_idx' (this bucket
    includes entries that were NaN-filtered when the fp16 cache was written —
    the fp16 index never contains NaN entries, so there's no need to re-scan).
    Returns (dataset, sid_list, drop_counts).
    """
    skeletons = []
    sid_list  = []
    missing = missing_fp16_idx = parse_failed = 0

    for sid in str_ids:
        doc = docs_by_id.get(sid)
        if doc is None:
            missing += 1
            continue
        if sid not in fp16_index:
            missing_fp16_idx += 1   # includes NaN-filtered entries
            continue
        result = build_graph_skeleton(doc.get("code", ""), label=int(doc.get("label", 0)))
        if result is None:
            parse_failed += 1
            continue
        edge_index, edge_attr, y, n_nodes = result
        skeletons.append((sid, edge_index, edge_attr, y, n_nodes))
        sid_list.append(sid)

    dataset = CodeBERTFP16MmapDataset(skeletons, data_path, fp16_index)
    drops = {
        "missing_db":       missing,
        "missing_fp16_idx": missing_fp16_idx,
        "parse_failed":     parse_failed,
    }
    return dataset, sid_list, drops


# ── inference ─────────────────────────────────────────────────────────────────

def _run_inference(model, dataset, batch_size: int):
    """Run forward pass over the whole dataset; return (labels, probs) as np arrays."""
    device = torch.device("cpu")
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    return np.array(all_labels), np.array(all_probs)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Graphault GNN checkpoint on the deduped test split."
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--features", required=True, choices=["onehot", "codebert"],
                        help="Feature type the checkpoint was trained with")
    parser.add_argument("--test-ids", default="data/splits/test_ids.json", type=Path,
                        help="Path to test_ids.json (default: data/splits/test_ids.json)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    test_ids_path   = args.test_ids

    if not checkpoint_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {checkpoint_path}")
    if not test_ids_path.exists():
        sys.exit(f"ERROR: test-ids file not found: {test_ids_path}")

    # ── 1. load test IDs — no re-sampling, no splitting ───────────────────────
    with open(test_ids_path) as f:
        str_ids = json.load(f)
    print(f"\nTest IDs loaded : {len(str_ids):,}  (from {test_ids_path})", flush=True)

    # ── 2. fetch code + labels from MongoDB ────────────────────────────────────
    print("Fetching docs from MongoDB...", flush=True)
    docs_by_id = _fetch_docs_for_str_ids(str_ids)
    print(f"  docs retrieved: {len(docs_by_id):,}\n", flush=True)

    # ── 3. build dataset ────────────────────────────────────────────────────────
    if args.features == "codebert":
        if not FP16_CACHE_DATA_PATH.exists():
            sys.exit(f"ERROR: fp16 cache not found: {FP16_CACHE_DATA_PATH}\n"
                     "Run train_deduped_codebert() once to produce it.")
        if not FP16_CACHE_INDEX_PATH.exists():
            sys.exit(f"ERROR: fp16 index not found: {FP16_CACHE_INDEX_PATH}")

        print(f"Loading fp16 index ({FP16_CACHE_INDEX_PATH})...", flush=True)
        with open(FP16_CACHE_INDEX_PATH) as f:
            fp16_index = json.load(f)
        print(f"  fp16 index entries: {len(fp16_index):,}\n", flush=True)

        print("Building CodeBERT dataset...", flush=True)
        dataset, sid_list, drops = _build_codebert_dataset(
            str_ids, docs_by_id, fp16_index, FP16_CACHE_DATA_PATH
        )
        n_loaded = len(dataset)
        n_dropped = len(str_ids) - n_loaded

        print(f"\n  IDs in test_ids.json : {len(str_ids):,}")
        print(f"  Graphs loaded        : {n_loaded:,}")
        print(f"  Dropped total        : {n_dropped:,}")
        print(f"    missing in MongoDB : {drops['missing_db']:,}")
        print(f"    missing in fp16 index (incl. NaN-filtered) : {drops['missing_fp16_idx']:,}")
        print(f"    parse failure      : {drops['parse_failed']:,}")

        # positive / negative counts from skeleton labels
        n_pos = sum(int(sk[3].item()) for sk in dataset.skeletons)
        in_channels = CODEBERT_DIM
        batch_size  = 64

    else:  # onehot
        print("Building one-hot dataset...", flush=True)
        dataset, sid_list, drops = _build_onehot_dataset(str_ids, docs_by_id)
        n_loaded = len(dataset)
        n_dropped = len(str_ids) - n_loaded

        print(f"\n  IDs in test_ids.json : {len(str_ids):,}")
        print(f"  Graphs loaded        : {n_loaded:,}")
        print(f"  Dropped total        : {n_dropped:,}")
        print(f"    missing in MongoDB : {drops['missing_db']:,}")
        print(f"    parse failure      : {drops['parse_failed']:,}")

        # positive / negative counts from graph labels
        n_pos = sum(int(g.y.item()) for g in dataset)
        in_channels = VOCAB_SIZE   # 89
        batch_size  = 128

    n_neg = n_loaded - n_pos
    random_baseline = n_pos / max(n_loaded, 1)

    print(f"\n  Positive (label=1) : {n_pos:,}")
    print(f"  Negative (label=0) : {n_neg:,}")
    print(f"  Random baseline PR-AUC : {random_baseline:.4f}", flush=True)

    # Sanity gate: flag if the number loaded is wildly off from the expected ~16K
    if not (10_000 <= n_loaded <= 20_000):
        sys.exit(
            f"\nSTOP: loaded {n_loaded:,} graphs — expected ~16,000. "
            "This is the 'reading wrong data' signal. "
            "Do NOT patch the test set or metric code to compensate."
        )

    # ── 4. load checkpoint ──────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {checkpoint_path}", flush=True)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pooling = "attention" if any(k.startswith("attn_pool") for k in state_dict) else "mean"
    model = CodeRiskGNN(in_channels, HIDDEN_DIM, 1, pooling=pooling)
    model.load_state_dict(state_dict)
    actual_in = model.conv1.in_channels
    print(f"  conv1.in_channels = {actual_in}  (89 = one-hot, 768 = CodeBERT)", flush=True)

    if actual_in != in_channels:
        sys.exit(
            f"\nSTOP: checkpoint has conv1.in_channels={actual_in} "
            f"but --features {args.features} expects {in_channels}. "
            "Checkpoint and feature flag are mismatched."
        )

    # ── 5. inference ────────────────────────────────────────────────────────────
    print("\nRunning inference...", flush=True)
    labels, probs = _run_inference(model, dataset, batch_size)

    # ── 6. metrics — verbatim from train.py:678-697 (codebert) / 310-330 (89dim)
    print("\n--- Evaluation on test split ---", flush=True)

    pr_auc       = average_precision_score(labels, probs)
    best_t, best_f1, best_p, best_r = _find_best_threshold(labels, probs)
    f1_at_half   = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    prec_at_half = precision_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    rec_at_half  = recall_score(labels, (probs >= 0.5).astype(int), zero_division=0)

    print(f"\n  PR-AUC (test)          : {pr_auc:.4f}")
    print(f"  Random baseline PR-AUC : {random_baseline:.4f}  (positive rate of test set)")
    print(f"  Uplift over random     : {pr_auc / random_baseline:.2f}x")
    print(f"\n  --- At F1-optimal threshold ({best_t:.4f}) ---")
    print(f"  F1        : {best_f1:.4f}")
    print(f"  Precision : {best_p:.4f}")
    print(f"  Recall    : {best_r:.4f}")
    print(f"\n  --- At threshold=0.5 ---")
    print(f"  F1        : {f1_at_half:.4f}")
    print(f"  Precision : {prec_at_half:.4f}")
    print(f"  Recall    : {rec_at_half:.4f}")
    print(f"\n  Checkpoint : {checkpoint_path}")
    print(f"  Features   : {args.features}  (conv1.in_channels={actual_in})")

    # ── 7. per-function predictions dump ────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{checkpoint_path.stem}_test_preds.json"
    preds_out = [
        {
            "function_id": sid,
            "true_label":  int(labels[i]),
            "risk_score":  round(float(probs[i]), 6),
        }
        for i, sid in enumerate(sid_list)
    ]
    with open(out_path, "w") as f:
        json.dump(preds_out, f, indent=2)
    print(f"\n  Per-function predictions -> {out_path}  ({len(preds_out):,} entries)")


if __name__ == "__main__":
    main()
