import os
import sys
import json
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'parser'))
from db import get_db
from gnn import CodeRiskGNN
from ast_graph_builder import build_ast_graph, VOCAB_SIZE

_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH         = _ROOT / "data" / "model.pt"
MODEL_DEDUPED_PATH = _ROOT / "data" / "model_deduped_89dim.pt"
SPLITS_DIR         = _ROOT / "data" / "splits"

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


if __name__ == "__main__":
    train_deduped()
