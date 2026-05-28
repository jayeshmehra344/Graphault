"""
Finds the optimal classification threshold from the validation set.
Uses the exact same 20K sample + func_name split as train.py (seed=42).
Saves PR curve to data/pr_curve.png and prints the best threshold.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves to file
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'parser'))
from db import get_db
from gnn import CodeRiskGNN
from ast_graph_builder import build_ast_graph, VOCAB_SIZE

ROOT       = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "data" / "model.pt"
PLOT_PATH  = ROOT / "data" / "pr_curve.png"

HIDDEN_DIM   = 64
BATCH_SIZE   = 128
SAMPLE_SIZE  = 20_000


# ── Reproduce exact load + split from train.py ───────────────────────────────

def load_dataset():
    db = get_db()
    print(f"Sampling {SAMPLE_SIZE} functions (seed=42, same as training)...", flush=True)
    all_ids = [d["_id"] for d in db["labeled_functions"].find({}, {"_id": 1})]
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(all_ids), size=min(SAMPLE_SIZE, len(all_ids)), replace=False)
    sample_ids = [all_ids[i] for i in chosen]
    docs = list(db["labeled_functions"].find(
        {"_id": {"$in": sample_ids}},
        {"func_name": 1, "code": 1, "label": 1, "_id": 0},
    ))
    print(f"Building AST graphs...", flush=True)
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
            print(f"  {i + 1}/{len(docs)} (skipped: {skipped})", flush=True)
    print(f"Graphs: {len(graphs)} | Skipped: {skipped}")
    return graphs, func_names


def func_name_split(graphs, func_names, train_ratio=0.8, seed=42):
    groups = defaultdict(list)
    for i, name in enumerate(func_names):
        groups[name].append(i)
    rng = np.random.default_rng(seed)
    all_names = np.array(list(groups.keys()))
    rng.shuffle(all_names)
    cut = int(train_ratio * len(all_names))
    val_names = set(all_names[cut:])
    val_idx = [i for n in val_names for i in groups[n]]
    return [graphs[i] for i in val_idx]


# ── Inference ─────────────────────────────────────────────────────────────────

def get_val_predictions(val_data):
    device = torch.device("cpu")
    model = CodeRiskGNN(VOCAB_SIZE, HIDDEN_DIM, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print(f"Model loaded from {MODEL_PATH}")

    loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_logits = model(batch.x, batch.edge_index)
            logits = global_mean_pool(node_logits, batch.batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().tolist())

    return np.array(all_labels), np.array(all_probs)


# ── Threshold search ──────────────────────────────────────────────────────────

def find_best_threshold(labels, probs):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve appends a final point with no threshold;
    # align arrays so each threshold corresponds to one (P, R) pair
    p, r, t = precision[:-1], recall[:-1], thresholds

    # F1 at each threshold (guard against P+R=0)
    denom = p + r
    f1_scores = np.where(denom > 0, 2 * p * r / denom, 0.0)

    best_idx = f1_scores.argmax()
    return float(t[best_idx]), float(f1_scores[best_idx]), float(p[best_idx]), float(r[best_idx])


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_pr_curve(labels, probs, best_threshold, best_f1, pr_auc):
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})")

    # Mark best-F1 threshold point
    # Find the point closest to best_threshold on the curve
    idx = np.searchsorted(thresholds, best_threshold)
    idx = min(idx, len(recall) - 2)  # stay in bounds
    ax.scatter(recall[idx], precision[idx], s=120, zorder=5, color="crimson",
               label=f"Best F1={best_f1:.3f} @ t={best_threshold:.3f}")

    # Random baseline
    pos_rate = labels.mean()
    ax.axhline(pos_rate, color="gray", linestyle="--", lw=1,
               label=f"Random baseline (P={pos_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Validation Set")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"PR curve saved to {PLOT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graphs, func_names = load_dataset()
    val_data = func_name_split(graphs, func_names)
    print(f"Val set: {len(val_data)} graphs\n")

    labels, probs = get_val_predictions(val_data)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    pr_auc = average_precision_score(labels, probs)
    print(f"Val  — positive: {n_pos} | negative: {n_neg} | PR-AUC: {pr_auc:.4f}")

    best_t, best_f1, best_p, best_r = find_best_threshold(labels, probs)
    print(f"\nOptimal threshold : {best_t:.4f}")
    print(f"  F1        : {best_f1:.4f}")
    print(f"  Precision : {best_p:.4f}")
    print(f"  Recall    : {best_r:.4f}")

    # Metrics at default 0.5 for comparison
    f1_at_half = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    print(f"\nF1 at threshold=0.5 : {f1_at_half:.4f}")
    print(f"F1 improvement      : +{best_f1 - f1_at_half:.4f}")

    plot_pr_curve(labels, probs, best_t, best_f1, pr_auc)

    print(f"\nUpdate RISK_THRESHOLD = {best_t:.4f} in src/api/app.py")
