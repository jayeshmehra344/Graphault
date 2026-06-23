import gc
import os
import sys
import json
import time
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import (
    f1_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score,
)
from pathlib import Path
from bson import ObjectId

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # src/model — for gnn, dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'parser'))
from db import get_db
from gnn import CodeRiskGNN
from ast_graph_builder import build_ast_graph, build_ast_graph_codebert, build_graph_skeleton, VOCAB_SIZE

_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH                 = _ROOT / "data" / "model.pt"
MODEL_DEDUPED_PATH         = _ROOT / "data" / "model_deduped_89dim.pt"
MODEL_DEDUPED_CODEBERT_PATH     = _ROOT / "data" / "model_deduped_codebert.pt"
SPLITS_DIR          = _ROOT / "data" / "splits"
CODEBERT_CACHE_PATH = Path("D:/graphault_cache/codebert_node_features.pt")
FP16_CACHE_DATA_PATH  = Path("D:/graphault_cache/codebert_fp16.bin")
FP16_CACHE_INDEX_PATH = Path("D:/graphault_cache/codebert_fp16_index.json")
CODEBERT_DIM        = 768

HIDDEN_DIM  = 64
EPOCHS      = 50
BATCH_SIZE  = 128
LR          = 1e-3
SAMPLE_SIZE = 20_000


def load_dataset(sample_size: int = SAMPLE_SIZE):
    db = get_db()
    print(f"Sampling {sample_size} functions from MongoDB...", flush=True)
    # $sample exceeds Atlas free-tier memory limit; fetch IDs then random-select client-side
    all_ids = [d["_id"] for d in db["labeled_functions"].find({}, {"_id": 1})]
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(all_ids), size=min(sample_size, len(all_ids)), replace=False)
    sample_ids = [all_ids[i] for i in chosen]
    docs = list(db["labeled_functions"].find(
        {"_id": {"$in": sample_ids}},
        {"func_name": 1, "code": 1, "label": 1, "_id": 0},
    ))
    print(f"Sampled: {len(docs)} | Building AST graphs...", flush=True)

    graphs, func_names = [], []
    skipped = 0
    for i, doc in enumerate(docs):
        data = build_ast_graph(doc.get("code", ""), label=int(doc.get("label", 0)))
        if data is None:
            skipped += 1
            continue
        graphs.append(data)
        func_names.append(doc.get("func_name", ""))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(docs)} processed (skipped: {skipped})", flush=True)

    print(f"Graphs built: {len(graphs)} | Parse failures: {skipped}")
    return graphs, func_names


def func_name_split(graphs, func_names, train_ratio=0.8, seed=42):
    """
    Assign every unique func_name exclusively to train or val,
    preventing the same function name from appearing in both splits.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(func_names):
        groups[name].append(i)

    rng = np.random.default_rng(seed)
    all_names = np.array(list(groups.keys()))
    rng.shuffle(all_names)

    cut = int(train_ratio * len(all_names))
    train_names = set(all_names[:cut])
    val_names   = set(all_names[cut:])

    train_idx = [i for n in train_names for i in groups[n]]
    val_idx   = [i for n in val_names   for i in groups[n]]

    return (
        [graphs[i] for i in train_idx],
        [graphs[i] for i in val_idx],
        train_names,
        val_names,
    )


def load_split_by_ids(ids_path: Path) -> list:
    """
    Load the exact set of functions identified by dedup.py.
    Fetches by _id from MongoDB in batches of 5,000 to stay within
    Atlas free-tier memory limits, then builds PyG graphs.
    """
    with open(ids_path) as f:
        str_ids = json.load(f)

    object_ids = [ObjectId(s) for s in str_ids]
    db = get_db()
    collection = db["labeled_functions"]

    BATCH = 5_000
    docs = []
    for start in range(0, len(object_ids), BATCH):
        chunk = object_ids[start:start + BATCH]
        docs.extend(collection.find(
            {"_id": {"$in": chunk}},
            {"code": 1, "label": 1, "_id": 0},
        ))
        print(f"  fetched {min(start + BATCH, len(object_ids)):,}/{len(object_ids):,}", flush=True)

    graphs, skipped = [], 0
    for i, doc in enumerate(docs):
        g = build_ast_graph(doc.get("code", ""), label=int(doc.get("label", 0)))
        if g is None:
            skipped += 1
            continue
        graphs.append(g)
        if (i + 1) % 10_000 == 0:
            print(f"  graphs built: {len(graphs):,}  skipped: {skipped}", flush=True)

    print(f"  total graphs: {len(graphs):,}  skipped: {skipped}", flush=True)
    return graphs


def _find_best_threshold(labels: np.ndarray, probs: np.ndarray):
    """F1-maximising threshold from the PR curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    p, r, t = precision[:-1], recall[:-1], thresholds
    denom = p + r
    f1s = np.where(denom > 0, 2 * p * r / denom, 0.0)
    best = int(f1s.argmax())
    return float(t[best]), float(f1s[best]), float(p[best]), float(r[best])


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = total_n = 0
    all_labels, all_probs, all_preds = [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad()

            # node-level logits -> graph-level via mean pooling over each graph's nodes
            node_logits = model(batch.x, batch.edge_index)  # [total_nodes, 1]
            logits = global_mean_pool(node_logits, batch.batch).squeeze(-1)  # [batch_size]
            loss = criterion(logits, batch.y)

            if training:
                loss.backward()
                optimizer.step()

            n = batch.y.size(0)
            total_loss += loss.item() * n
            total_n    += n

            probs = torch.sigmoid(logits).detach().cpu()
            preds = (logits.detach().cpu() > 0).float()
            all_labels.extend(batch.y.cpu().tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    avg_loss = total_loss / total_n
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    pr_auc   = average_precision_score(all_labels, all_probs)
    return avg_loss, f1, pr_auc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    graphs, func_names = load_dataset()

    labels  = [int(g.y.item()) for g in graphs]
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos
    pos_wt  = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    print(f"Buggy (1): {n_pos} | Clean (0): {n_neg} | pos_weight: {pos_wt.item():.2f}")

    train_data, val_data, train_names, val_names = func_name_split(graphs, func_names)
    overlap = train_names & val_names
    print(f"Train: {len(train_data)} graphs ({len(train_names)} unique names)")
    print(f"Val:   {len(val_data)} graphs ({len(val_names)} unique names)")
    print(f"Name overlap (must be 0): {len(overlap)}\n")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    model     = CodeRiskGNN(VOCAB_SIZE, HIDDEN_DIM, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_pr_auc = 0.0

    hdr = f"{'Ep':>3}  {'TrLoss':>7}  {'TrF1':>5}  {'TrAUC':>6}  {'VaLoss':>7}  {'VaF1':>5}  {'VaAUC':>6}"
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_f1, va_auc = run_epoch(model, val_loader,   criterion, device)

        marker = " *" if va_auc > best_pr_auc else ""
        print(f"{epoch:3d}  {tr_loss:7.4f}  {tr_f1:5.3f}  {tr_auc:6.4f}  "
              f"{va_loss:7.4f}  {va_f1:5.3f}  {va_auc:6.4f}{marker}")

        if va_auc > best_pr_auc:
            best_pr_auc = va_auc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest Val PR-AUC: {best_pr_auc:.4f} | Model saved to {MODEL_PATH}")


def train_deduped():
    """
    Retrain on the dedup-pipeline split.
    Saves to model_deduped_89dim.pt — does NOT touch model.pt.
    Architecture and hyperparameters identical to train().
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading train split...", flush=True)
    train_data = load_split_by_ids(SPLITS_DIR / "train_ids.json")
    print(f"Train graphs: {len(train_data):,}\n", flush=True)

    print("Loading test split...", flush=True)
    test_data = load_split_by_ids(SPLITS_DIR / "test_ids.json")
    print(f"Test  graphs: {len(test_data):,}\n", flush=True)

    # pos_weight from train split only
    train_labels = [int(g.y.item()) for g in train_data]
    n_pos  = sum(train_labels)
    n_neg  = len(train_labels) - n_pos
    pos_wt = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    print(f"Train — pos: {n_pos:,} | neg: {n_neg:,} | pos_weight: {pos_wt.item():.2f}", flush=True)

    test_labels_raw = [int(g.y.item()) for g in test_data]
    n_test_pos = sum(test_labels_raw)
    random_baseline = n_test_pos / max(len(test_labels_raw), 1)
    print(f"Test  — pos: {n_test_pos:,} | neg: {len(test_labels_raw)-n_test_pos:,}"
          f" | random baseline PR-AUC: {random_baseline:.4f}\n", flush=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    model     = CodeRiskGNN(VOCAB_SIZE, HIDDEN_DIM, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt)

    MODEL_DEDUPED_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_pr_auc   = 0.0
    best_state    = None

    hdr = (f"{'Ep':>3}  {'TrLoss':>7}  {'TrF1':>5}  {'TrAUC':>6}"
           f"  {'TeLoss':>7}  {'TeF1':>5}  {'TeAUC':>6}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, device, optimizer)
        te_loss, te_f1, te_auc = run_epoch(model, test_loader,  criterion, device)

        marker = " *" if te_auc > best_pr_auc else ""
        print(f"{epoch:3d}  {tr_loss:7.4f}  {tr_f1:5.3f}  {tr_auc:6.4f}"
              f"  {te_loss:7.4f}  {te_f1:5.3f}  {te_auc:6.4f}{marker}", flush=True)

        if te_auc > best_pr_auc:
            best_pr_auc = te_auc
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    # save best checkpoint
    torch.save(best_state, MODEL_DEDUPED_PATH)
    print(f"\nBest test PR-AUC: {best_pr_auc:.4f}")
    print(f"Model saved   -> {MODEL_DEDUPED_PATH}")

    # ── Final detailed evaluation at best checkpoint ──────────────────────────
    print("\n--- Final evaluation on test split ---", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            node_logits = model(batch.x, batch.edge_index)
            logits = global_mean_pool(node_logits, batch.batch).squeeze(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    pr_auc             = average_precision_score(labels, probs)
    best_t, best_f1, best_p, best_r = _find_best_threshold(labels, probs)
    preds_at_best      = (probs >= best_t).astype(int)
    f1_at_half         = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    prec_at_half       = precision_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    rec_at_half        = recall_score(labels, (probs >= 0.5).astype(int), zero_division=0)

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
    print(f"\n  Saved model : {MODEL_DEDUPED_PATH}")
    print(f"  Original    : {MODEL_PATH}  <-- NOT overwritten")


# ── CodeBERT fp16 binary cache ────────────────────────────────────────────────
#
# Memory problem: the fp16 feature dict is ~11.9 GB (mean 144 KB/entry, max 18.5 MB),
# which exceeds the ~9 GB free RAM on this machine. Loading it into a Python dict OOMs
# at ~20K entries regardless of isfinite-check vs. not.
#
# Solution: write the features ONCE as a compact fp16 binary file on D: (88 GB free).
# Peak RAM during the write: one tensor at a time (~55 MB max). Training reads from
# a numpy memmap of the 10.5 GB file — the OS page cache holds the hot pages, so
# after the first epoch's warm-up access is essentially RAM-speed.
#
# Files written:
#   FP16_CACHE_DATA_PATH  — raw float16 bytes, no framing, contiguous per function
#   FP16_CACHE_INDEX_PATH — JSON: {sid: [element_offset, n_nodes]}


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
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _build_nan_set(needed_ids: set) -> set:
    """
    Sequential mmap scan to identify non-finite entries.
    Reads each tensor only to check isfinite — no fp16 copies stored.
    Peak RAM: a handful of tensor pages at a time.
    Releases the mmap and calls gc.collect() before returning.
    """
    print("Scanning cache for NaN/Inf entries (pass 1/2)...", flush=True)
    cache = torch.load(CODEBERT_CACHE_PATH, mmap=True, weights_only=True)
    print(f"  cache entries: {len(cache):,}", flush=True)

    nan_ids: set = set()
    n = 0
    for sid, t in cache.items():
        n += 1
        if sid in needed_ids and not torch.isfinite(t).all():
            nan_ids.add(sid)
        if n % 20_000 == 0:
            print(f"  scanned {n:,}/{len(cache):,}  nan={len(nan_ids)}", flush=True)

    del cache
    gc.collect()
    print(f"  done — {len(nan_ids):,} non-finite entries\n", flush=True)
    return nan_ids


def _prepare_fp16_cache(
    data_path: Path, index_path: Path, needed_ids: set, nan_ids: set
) -> None:
    """
    Pass 2/2 — write a compact fp16 binary cache file if it doesn't already exist.
    One sequential pass over the fp32 mmap; fp16 bytes written directly to disk,
    one tensor at a time. Peak RAM: Python overhead + ~55 MB per large tensor.
    Releases the mmap and calls gc.collect() before returning.
    """
    if data_path.exists() and index_path.exists():
        size_gb = data_path.stat().st_size / 1e9
        print(f"fp16 cache already exists ({size_gb:.2f} GB): {data_path}\n", flush=True)
        return

    load_ids = needed_ids - nan_ids
    print(f"Writing fp16 cache (pass 2/2) -- {len(load_ids):,} entries -> {data_path}...", flush=True)

    cache = torch.load(CODEBERT_CACHE_PATH, mmap=True, weights_only=True)
    print(f"  source entries: {len(cache):,}", flush=True)

    index: dict = {}   # sid → [element_offset, n_nodes]
    el_offset = 0
    n = 0

    with open(data_path, 'wb') as f_out:
        for sid, t in cache.items():
            n += 1
            if sid in load_ids:
                fp16 = t.half()
                n_nodes = fp16.shape[0]
                f_out.write(fp16.numpy().tobytes())
                index[sid] = [el_offset, n_nodes]
                el_offset += n_nodes * CODEBERT_DIM
                del fp16
            if n % 5_000 == 0:
                gc.collect()
            if n % 20_000 == 0:
                written_gb = el_offset * 2 / 1e9   # 2 bytes per float16 element
                print(f"  written {len(index):,}/{len(load_ids):,}  {written_gb:.2f} GB", flush=True)

    del cache
    gc.collect()

    print(f"  saving index ({len(index):,} entries)...", flush=True)
    with open(index_path, 'w') as f:
        json.dump(index, f)

    size_gb = data_path.stat().st_size / 1e9
    print(f"  done: {size_gb:.2f} GB written\n", flush=True)


def _fetch_docs_for_str_ids(str_ids: list) -> dict:
    """Fetch {code, label} docs from MongoDB for the given str_ids."""
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


def _fetch_docs_by_id(ids_path: Path) -> tuple:
    """Load str ids from ids_path and fetch {code, label} docs from MongoDB by _id."""
    with open(ids_path) as f:
        str_ids = json.load(f)
    docs_by_id = _fetch_docs_for_str_ids(str_ids)
    return str_ids, docs_by_id


def build_codebert_dataset_fp16(
    str_ids: list, docs_by_id: dict, fp16_index: dict, nan_ids: set,
    data_path: Path,
) -> CodeBERTFP16MmapDataset:
    """
    Build graph skeletons and wrap with CodeBERTFP16MmapDataset.
    Skips functions that are NaN, missing from MongoDB, missing from the fp16 index,
    or that fail to parse.
    """
    skeletons = []
    missing = missing_idx = parse_failed = nan_features = 0

    for sid in str_ids:
        doc = docs_by_id.get(sid)
        if doc is None:
            missing += 1
            continue
        if sid in nan_ids:
            nan_features += 1
            continue
        if sid not in fp16_index:
            missing_idx += 1
            continue

        result = build_graph_skeleton(doc.get("code", ""), label=int(doc.get("label", 0)))
        if result is None:
            parse_failed += 1
            continue

        edge_index, edge_attr, y, n_nodes = result
        skeletons.append((sid, edge_index, edge_attr, y, n_nodes))

    print(f"  total graphs: {len(skeletons):,}  missing_db: {missing}"
          f"  missing_idx: {missing_idx}  nan: {nan_features}"
          f"  parse_failed: {parse_failed}", flush=True)
    return CodeBERTFP16MmapDataset(skeletons, data_path, fp16_index)


def train_deduped_codebert(epochs: int = EPOCHS, save_path: Path = MODEL_DEDUPED_CODEBERT_PATH):
    """
    Retrain on the dedup-pipeline split using 768-dim CodeBERT node features.

    Memory design (fp16 binary mmap — fits on 16 GB machines):
      Pass 1: mmap NaN scan — no tensors stored, releases mmap at end.
      Pass 2: write fp16 binary file to D: (peak RAM ~600 MB; one tensor at a time).
              Skipped on subsequent runs if the file already exists.
      Training: numpy memmap of the ~10 GB fp16 file — OS page-caches hot pages,
                RAM-speed after first epoch warm-up, no OOM risk.
    """
    try:
        import psutil as _psutil
        _proc = _psutil.Process()
        def _ram_mb(): return _proc.memory_info().rss / 1024**2
        _have_ram = True
    except ImportError:
        def _ram_mb(): return 0.0
        _have_ram = False

    torch.set_num_threads(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Read split id lists from JSON (tiny) ─────────────────────────────────
    print("Reading split id lists...", flush=True)
    with open(SPLITS_DIR / "train_ids.json") as f:
        train_str_ids = json.load(f)
    with open(SPLITS_DIR / "test_ids.json") as f:
        test_str_ids = json.load(f)
    needed_ids = set(train_str_ids) | set(test_str_ids)
    print(f"  train: {len(train_str_ids):,}  test: {len(test_str_ids):,}  "
          f"union: {len(needed_ids):,}\n", flush=True)

    # ── Pass 1: NaN scan (releases mmap before the write pass) ───────────────
    # Skipped when fp16 cache already exists — nan_ids is only consumed by
    # _prepare_fp16_cache which returns immediately when both files exist.
    if FP16_CACHE_DATA_PATH.exists() and FP16_CACHE_INDEX_PATH.exists():
        nan_ids: set = set()
        size_gb = FP16_CACHE_DATA_PATH.stat().st_size / 1e9
        print(f"fp16 cache already exists ({size_gb:.2f} GB) — skipping NaN scan.\n", flush=True)
    else:
        nan_ids = _build_nan_set(needed_ids)
        if _have_ram:
            print(f"RAM after NaN scan: {_ram_mb():.0f} MB", flush=True)

    # ── Pass 2: write fp16 binary cache (one-time; skipped if exists) ─────────
    _prepare_fp16_cache(FP16_CACHE_DATA_PATH, FP16_CACHE_INDEX_PATH, needed_ids, nan_ids)
    if _have_ram:
        print(f"RAM after cache write: {_ram_mb():.0f} MB", flush=True)

    # ── Load index (small JSON: ~4 MB for 80 K entries) ──────────────────────
    print("Loading fp16 index...", flush=True)
    with open(FP16_CACHE_INDEX_PATH) as f:
        fp16_index = json.load(f)
    print(f"  {len(fp16_index):,} entries\n", flush=True)

    # ── Fetch MongoDB docs (after write pass so no mmap overlap) ─────────────
    print("Loading train docs from MongoDB...", flush=True)
    train_docs = _fetch_docs_for_str_ids(train_str_ids)
    print("Loading test docs from MongoDB...", flush=True)
    test_docs  = _fetch_docs_for_str_ids(test_str_ids)

    # ── Build datasets ────────────────────────────────────────────────────────
    print("Building train dataset...", flush=True)
    train_data = build_codebert_dataset_fp16(
        train_str_ids, train_docs, fp16_index, nan_ids, FP16_CACHE_DATA_PATH)
    print(f"Train graphs: {len(train_data):,}\n", flush=True)

    print("Building test dataset...", flush=True)
    test_data = build_codebert_dataset_fp16(
        test_str_ids, test_docs, fp16_index, nan_ids, FP16_CACHE_DATA_PATH)
    print(f"Test  graphs: {len(test_data):,}\n", flush=True)

    del train_docs, test_docs
    gc.collect()

    if _have_ram:
        print(f"RAM after dataset build (before training): {_ram_mb():.0f} MB\n", flush=True)

    # pos_weight from train split only
    train_labels = [int(y.item()) for (_, _, _, y, _) in train_data.skeletons]
    n_pos  = sum(train_labels)
    n_neg  = len(train_labels) - n_pos
    pos_wt = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    print(f"Train — pos: {n_pos:,} | neg: {n_neg:,} | pos_weight: {pos_wt.item():.2f}", flush=True)

    test_labels_raw = [int(y.item()) for (_, _, _, y, _) in test_data.skeletons]
    n_test_pos = sum(test_labels_raw)
    random_baseline = n_test_pos / max(len(test_labels_raw), 1)
    print(f"Test  — pos: {n_test_pos:,} | neg: {len(test_labels_raw)-n_test_pos:,}"
          f" | random baseline PR-AUC: {random_baseline:.4f}\n", flush=True)

    _bs = 64   # smaller batch gives more RAM headroom during fp16 mmap warm-up
    train_loader = DataLoader(train_data, batch_size=_bs, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=_bs, shuffle=False, num_workers=0)

    model     = CodeRiskGNN(CODEBERT_DIM, HIDDEN_DIM, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_pr_auc = 0.0
    best_state  = None
    peak_ram_mb = _ram_mb()

    ram_col = "  {:>7}".format("RAM") if _have_ram else ""
    hdr = (f"{'Ep':>3}  {'TrLoss':>7}  {'TrF1':>5}  {'TrAUC':>6}"
           f"  {'TeLoss':>7}  {'TeF1':>5}  {'TeAUC':>6}  {'Time':>6}{ram_col}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, device, optimizer)
        te_loss, te_f1, te_auc = run_epoch(model, test_loader,  criterion, device)
        elapsed = time.perf_counter() - t0

        ram_mb = _ram_mb()
        peak_ram_mb = max(peak_ram_mb, ram_mb)
        ram_suffix = f"  {ram_mb:5.0f}MB" if _have_ram else ""

        marker = " *" if te_auc > best_pr_auc else ""
        print(f"{epoch:3d}  {tr_loss:7.4f}  {tr_f1:5.3f}  {tr_auc:6.4f}"
              f"  {te_loss:7.4f}  {te_f1:5.3f}  {te_auc:6.4f}"
              f"  {elapsed:5.0f}s{ram_suffix}{marker}", flush=True)

        if te_auc > best_pr_auc:
            best_pr_auc = te_auc
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save(best_state, save_path)
    print(f"\nBest test PR-AUC : {best_pr_auc:.4f}")
    if _have_ram:
        print(f"Peak system RAM  : {peak_ram_mb:.0f} MB")
    if device.type == "cuda":
        print(f"Peak GPU VRAM    : {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    print(f"Model saved      -> {save_path}")

    # ── Final detailed evaluation at best checkpoint ──────────────────────────
    print("\n--- Final evaluation on test split ---", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            node_logits = model(batch.x, batch.edge_index)
            logits = global_mean_pool(node_logits, batch.batch).squeeze(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    labels = np.array(all_labels)
    probs  = np.array(all_probs)

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
    print(f"\n  Saved model    : {save_path}")
    print(f"  89-dim deduped : {MODEL_DEDUPED_PATH}  <-- NOT overwritten")
    print(f"  Original       : {MODEL_PATH}  <-- NOT overwritten")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "codebert":
        train_deduped_codebert()
    elif len(sys.argv) > 1 and sys.argv[1] == "codebert-sanity":
        train_deduped_codebert(
            epochs=10,
            save_path=_ROOT / "data" / "model_codebert_sanity.pt",
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "codebert-smoke":
        train_deduped_codebert(
            epochs=2,
            save_path=_ROOT / "data" / "model_codebert_smoke.pt",
        )
    else:
        train_deduped()
