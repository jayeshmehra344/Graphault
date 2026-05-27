import json
import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
from db import get_db

OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "training_data.json"


def combine(seed=42):
    db = get_db()
    collection = db["labeled_functions"]

    print("Reading labeled_functions from MongoDB...")
    docs = list(collection.find({}, {"_id": 0, "func_name": 1, "label": 1, "source": 1}))
    print(f"  total documents : {len(docs)}")

    # Deduplicate by func_name — risky entries take priority over safe ones
    # so sort label=1 first before scanning
    docs.sort(key=lambda d: d.get("label", 0), reverse=True)
    seen = set()
    unique = []
    for doc in docs:
        name = doc.get("func_name", "").strip()
        if name and name not in seen:
            seen.add(name)
            unique.append(doc)
    print(f"  after dedup     : {len(unique)}")

    risky = [d for d in unique if d.get("label") == 1]
    safe  = [d for d in unique if d.get("label") == 0]
    print(f"  risky (label=1) : {len(risky)}")
    print(f"  safe  (label=0) : {len(safe)}")

    # Balance to equal class sizes
    n = min(len(risky), len(safe))
    random.seed(seed)
    balanced = random.sample(risky, n) + random.sample(safe, n)
    random.shuffle(balanced)
    print(f"  balanced total  : {len(balanced)}  ({n} each class)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(balanced, f, indent=2)
    print(f"\nSaved {len(balanced)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    combine()
