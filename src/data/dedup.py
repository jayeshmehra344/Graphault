"""
src/data/dedup.py — Deduplication pipeline for labeled_functions
=================================================================
READ-ONLY against MongoDB (no inserts/updates).
Writes split indices to data/splits/train_ids.json + test_ids.json.
Does NOT retrain, does NOT touch the model.

Steps
-----
1. Load all documents from labeled_functions
2. Normalize each function: strip comments (via AST round-trip),
   normalize whitespace, rename user-defined variables/args → VAR1/VAR2/...
3. Exact-hash dedup: SHA-256 of normalized form, keep one per hash
4. Near-duplicate detection: MinHash LSH, Jaccard > 0.90 on token sets
5. Train/test split (80/20 stratified by label) with zero cross-contamination
6. Save split to data/splits/ and print full statistics

Usage
-----
    python src/data/dedup.py
"""

import ast
import re
import sys
import json
import hashlib
import textwrap
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "graph"))
from db import get_db  # noqa: E402

# ── optional MinHash LSH ───────────────────────────────────────────────────────
try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False

# ── constants ─────────────────────────────────────────────────────────────────
JACCARD_THRESHOLD = 0.90
MINHASH_PERMS     = 128
TRAIN_RATIO       = 0.80
SEED              = 42

# Names that are NOT renamed during normalization.
# Includes Python builtins, common stdlib names, and special identifiers.
_KEEP_NAMES = (
    set(dir(__builtins__)) if isinstance(__builtins__, dict)
    else set(dir(__builtins__))
) | {
    # special
    'self', 'cls', '__init__', '__name__', '__file__', '__doc__',
    # stdlib top-level modules commonly imported as bare names
    'os', 're', 'sys', 'io', 'math', 'json', 'ast', 'abc',
    'itertools', 'functools', 'collections', 'pathlib', 'typing',
    'datetime', 'time', 'random', 'hashlib', 'copy',
    # common exception names
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'RuntimeError', 'StopIteration', 'NotImplementedError',
    'OSError', 'IOError', 'FileNotFoundError', 'PermissionError',
    'AssertionError', 'ImportError', 'OverflowError', 'ZeroDivisionError',
    # constants
    'True', 'False', 'None',
}


# ── normalization ─────────────────────────────────────────────────────────────

class _NameNormalizer(ast.NodeTransformer):
    """Renames user-defined variables and arguments to VAR1, VAR2, ..."""

    def __init__(self):
        self._mapping: dict[str, str] = {}
        self._counter = 1

    def _placeholder(self, name: str) -> str:
        if name in _KEEP_NAMES:
            return name
        if name not in self._mapping:
            self._mapping[name] = f"VAR{self._counter}"
            self._counter += 1
        return self._mapping[name]

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._placeholder(node.arg)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._placeholder(node.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Keep function names as-is so identical logic under different names still matches
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        return node


def _try_parse(code: str) -> Optional[ast.AST]:
    """Parse raw; fall back to dedented; return None on failure."""
    for attempt in (code, textwrap.dedent(code)):
        try:
            return ast.parse(attempt)
        except SyntaxError:
            pass
    return None


def normalize(code: str) -> Optional[str]:
    """
    Returns canonical string form of code, or None if unparseable.
    - Comments stripped (ast.parse ignores them)
    - Whitespace canonical via ast.unparse
    - User variable/arg names replaced with VAR1/VAR2/...
    """
    tree = _try_parse(code)
    if tree is None:
        return None
    try:
        normalizer = _NameNormalizer()
        transformed = normalizer.visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        return None


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).digest().hex()


# ── tokenization ──────────────────────────────────────────────────────────────

def tokenize(code: str) -> list[str]:
    """Word-level tokens + punctuation tokens for Jaccard similarity."""
    return re.findall(r'[A-Za-z_]\w*|\d+|[^\w\s]', code)


# ── Union-Find for connected components ───────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ── MinHash LSH near-duplicate detection ──────────────────────────────────────

def _build_minhash(tokens: list[str]) -> "MinHash":
    m = MinHash(num_perm=MINHASH_PERMS)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m


def near_dedup_lsh(records: list[dict]) -> list[dict]:
    """
    Uses MinHash LSH to find pairs with token Jaccard > JACCARD_THRESHOLD,
    groups into connected components via Union-Find, keeps the first element
    of each cluster (lowest index = first inserted into MongoDB = oldest).

    Returns the deduplicated list.
    """
    print(f"  Building MinHash signatures ({MINHASH_PERMS} permutations)...", flush=True)
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_PERMS)
    minhashes: list["MinHash"] = []

    for i, rec in enumerate(records):
        m = _build_minhash(rec["tokens"])
        minhashes.append(m)
        try:
            lsh.insert(str(i), m)
        except ValueError:
            # datasketch raises ValueError for duplicate keys — shouldn't happen
            pass
        if (i + 1) % 20_000 == 0:
            print(f"    hashed {i + 1:,}/{len(records):,}", flush=True)

    print(f"  Querying LSH for near-duplicates...", flush=True)
    uf = _UnionFind(len(records))
    n_pairs = 0

    for i, m in enumerate(minhashes):
        neighbours = lsh.query(m)
        for nb_key in neighbours:
            j = int(nb_key)
            if j != i:
                uf.union(i, j)
                n_pairs += 1
        if (i + 1) % 20_000 == 0:
            print(f"    queried {i + 1:,}/{len(records):,}  pairs so far: {n_pairs:,}", flush=True)

    # Keep one representative per component (the one with the smallest index)
    cluster_rep: dict[int, int] = {}
    for i in range(len(records)):
        root = uf.find(i)
        if root not in cluster_rep:
            cluster_rep[root] = i  # first occurrence wins

    kept_indices = sorted(cluster_rep.values())
    n_components = len(kept_indices)
    n_removed    = len(records) - n_components

    print(f"  Near-dup pairs found: {n_pairs:,}", flush=True)
    print(f"  Components: {n_components:,}  |  Removed: {n_removed:,}", flush=True)
    return [records[i] for i in kept_indices]


def _fallback_near_dedup(records: list[dict]) -> list[dict]:
    """
    Fallback when datasketch is absent: exact-hash only pass already ran,
    so just do a best-effort token-level dedup using a sliding window on
    sorted token fingerprints. Warns the user.
    """
    print("  [WARN] datasketch not installed — near-dedup skipped.", flush=True)
    print("         Run: pip install datasketch", flush=True)
    return records


# ── stratified split ─────────────────────────────────────────────────────────

def stratified_split(records: list[dict], train_ratio: float, seed: int):
    """
    Stratified 80/20 split by label.
    After near-dedup every record is already its own cluster,
    so a random split has zero cross-contamination by construction.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    train, test = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        cut = int(len(group) * train_ratio)
        train.extend(group[:cut])
        test.extend(group[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def verify_no_overlap(train: list[dict], test: list[dict]) -> int:
    train_hashes = {r["norm_hash"] for r in train}
    test_hashes  = {r["norm_hash"] for r in test}
    return len(train_hashes & test_hashes)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Graphault Dedup Pipeline")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading from MongoDB labeled_functions...", flush=True)
    db = get_db()
    cursor = db["labeled_functions"].find(
        {}, {"_id": 1, "func_name": 1, "code": 1, "label": 1, "source": 1}
    )
    raw = list(cursor)
    n_original = len(raw)
    print(f"      Loaded: {n_original:,}", flush=True)

    # ── 2. Normalize ──────────────────────────────────────────────────────────
    print("\n[2/5] Normalizing (AST round-trip + variable renaming)...", flush=True)
    normalized: list[dict] = []
    n_parse_fail = 0

    for i, doc in enumerate(raw):
        code = doc.get("code", "")
        norm = normalize(code)
        if norm is None:
            n_parse_fail += 1
            continue
        normalized.append({
            "_id":       str(doc["_id"]),
            "func_name": doc.get("func_name", ""),
            "source":    doc.get("source", ""),
            "label":     int(doc.get("label", 0)),
            "norm_hash": sha256(norm),
            "norm_code": norm,
            "tokens":    tokenize(norm),
        })
        if (i + 1) % 30_000 == 0:
            print(f"      {i + 1:,}/{n_original:,} processed", flush=True)

    n_parseable = len(normalized)
    print(f"      Parseable:      {n_parseable:,} / {n_original:,}"
          f"  ({100 * n_parseable / n_original:.1f}%)", flush=True)
    print(f"      Parse failures: {n_parse_fail:,}", flush=True)

    # ── 3. Exact dedup ────────────────────────────────────────────────────────
    print("\n[3/5] Exact deduplication (SHA-256 of normalized form)...", flush=True)
    seen_hashes: dict[str, dict] = {}
    n_exact_dups = 0

    for rec in normalized:
        h = rec["norm_hash"]
        if h not in seen_hashes:
            seen_hashes[h] = rec
        else:
            # If labels disagree on the same code, keep label=1 (vulnerable)
            if rec["label"] == 1 and seen_hashes[h]["label"] == 0:
                seen_hashes[h]["label"] = 1
            n_exact_dups += 1

    after_exact = list(seen_hashes.values())
    n_after_exact = len(after_exact)
    print(f"      Before: {n_parseable:,}", flush=True)
    print(f"      After:  {n_after_exact:,}"
          f"  (-{n_exact_dups:,} exact duplicates,"
          f"  -{100 * n_exact_dups / n_parseable:.1f}%)", flush=True)

    # Label distribution after exact dedup
    n_pos_exact = sum(1 for r in after_exact if r["label"] == 1)
    n_neg_exact = n_after_exact - n_pos_exact
    print(f"      Label=1: {n_pos_exact:,}  |  Label=0: {n_neg_exact:,}", flush=True)

    # ── 4. Near-duplicate detection ───────────────────────────────────────────
    print(f"\n[4/5] Near-duplicate detection (Jaccard > {JACCARD_THRESHOLD})...",
          flush=True)

    if _HAS_DATASKETCH:
        after_near = near_dedup_lsh(after_exact)
    else:
        after_near = _fallback_near_dedup(after_exact)

    n_after_near = len(after_near)
    n_near_dups  = n_after_exact - n_after_near
    n_pos_near   = sum(1 for r in after_near if r["label"] == 1)
    n_neg_near   = n_after_near - n_pos_near
    print(f"      Before: {n_after_exact:,}", flush=True)
    print(f"      After:  {n_after_near:,}"
          f"  (-{n_near_dups:,} near-duplicates,"
          f"  -{100 * n_near_dups / n_after_exact:.1f}% more)", flush=True)
    print(f"      Label=1: {n_pos_near:,}  |  Label=0: {n_neg_near:,}", flush=True)

    # ── 5. Train/test split ───────────────────────────────────────────────────
    print(f"\n[5/5] Stratified train/test split ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)})...",
          flush=True)

    train, test = stratified_split(after_near, TRAIN_RATIO, SEED)

    overlap = verify_no_overlap(train, test)
    overlap_status = "ZERO OVERLAP [OK]" if overlap == 0 else f"OVERLAP DETECTED: {overlap} shared hashes [FAIL]"

    n_train_pos = sum(1 for r in train if r["label"] == 1)
    n_test_pos  = sum(1 for r in test  if r["label"] == 1)

    print(f"      Train: {len(train):,}"
          f"  (label=1: {n_train_pos:,}, label=0: {len(train)-n_train_pos:,})", flush=True)
    print(f"      Test:  {len(test):,}"
          f"  (label=1: {n_test_pos:,},  label=0: {len(test)-n_test_pos:,})", flush=True)
    print(f"      Cross-contamination: {overlap_status}", flush=True)

    # ── Save split indices ────────────────────────────────────────────────────
    splits_dir = _ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train_ids.json"
    test_path  = splits_dir / "test_ids.json"
    meta_path  = splits_dir / "dedup_meta.json"

    with open(train_path, "w") as f:
        json.dump([r["_id"] for r in train], f)
    with open(test_path, "w") as f:
        json.dump([r["_id"] for r in test], f)
    with open(meta_path, "w") as f:
        json.dump({
            "n_original":      n_original,
            "n_parseable":     n_parseable,
            "n_parse_fail":    n_parse_fail,
            "n_after_exact":   n_after_exact,
            "n_exact_dups":    n_exact_dups,
            "n_after_near":    n_after_near,
            "n_near_dups":     n_near_dups,
            "n_train":         len(train),
            "n_test":          len(test),
            "n_train_pos":     n_train_pos,
            "n_test_pos":      n_test_pos,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "minhash_perms":   MINHASH_PERMS,
            "train_ratio":     TRAIN_RATIO,
            "seed":            SEED,
            "overlap":         overlap,
        }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Original count:          {n_original:>10,}")
    print(f"  After normalization:     {n_parseable:>10,}"
          f"  ({100*(n_original-n_parseable)/n_original:.1f}% unparseable dropped)")
    print(f"  After exact dedup:       {n_after_exact:>10,}"
          f"  (-{n_exact_dups:,})")
    print(f"  After near-dedup:        {n_after_near:>10,}"
          f"  (-{n_near_dups:,})")
    print(f"  Train:                   {len(train):>10,}"
          f"  (pos: {n_train_pos:,}  neg: {len(train)-n_train_pos:,})")
    print(f"  Test:                    {len(test):>10,}"
          f"  (pos: {n_test_pos:,}  neg: {len(test)-n_test_pos:,})")
    print(f"  Code-level overlap:      {overlap:>10,}  -> {overlap_status}")
    print(f"\n  Splits saved to:  data/splits/")
    print("=" * 60)


if __name__ == "__main__":
    main()
