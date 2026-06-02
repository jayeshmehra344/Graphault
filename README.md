# Graphault

A Graph Neural Network that detects vulnerability risk in Python functions by reasoning over their Abstract Syntax Tree structure — not pattern matching, not keyword search, but learned graph topology.

**Live:**
- API — `http://43.205.146.154` ([Swagger UI](http://43.205.146.154/docs))
- Frontend — `http://graphault-frontend.s3-website.ap-south-1.amazonaws.com`

---

## How it works

Most static analysis tools scan for known patterns. Graphault converts each Python function into a graph and lets a GNN learn what risky code *looks like structurally*, across thousands of examples drawn from real CVEs and bug datasets.

**Per-function pipeline:**

```
Python function (source code)
        │
        ▼
┌───────────────────┐
│   AST Parser      │  Python's ast module
│  (ast_graph_      │  Handles indented snippets,
│   builder.py)     │  no-def code fragments
└────────┬──────────┘
         │
         ▼  Graph: N nodes, E edges
    ┌─────────────────────────────────────────┐
    │  Nodes  — one per AST node              │
    │           feature: 89-dim one-hot       │
    │           of node type (Call, BinOp…)   │
    │                                         │
    │  Edges  — 3 types:                      │
    │    0: parent → child  (structural)      │
    │    1: child  → parent (structural)      │
    │    2: stmt   → next   (control flow)    │
    └────────────────┬────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   CodeRiskGNN         │
         │                       │
         │  GCNConv(89 → 64)     │
         │  ReLU + Dropout(0.3)  │
         │  GCNConv(64 → 64)     │
         │  ReLU + Dropout(0.3)  │
         │  GCNConv(64 →  1)     │
         │                       │
         │  mean-pool nodes      │
         │  → sigmoid → score    │
         └──────────┬────────────┘
                    │
                    ▼
         risk score 0..1  +  label (threshold 0.5417)
```

**Saliency (the `/explain` endpoint):** gradient of the output score w.r.t. each node's input features — the L2 norm per node is its contribution score. No extra dependencies; swap in PyG's `GNNExplainer` later without changing the API contract.

**Approach:** Inspired by [Devign (Zhou et al., 2019)](https://arxiv.org/abs/1909.03496) and [ReVeal (Chakraborty et al., 2021)](https://arxiv.org/abs/2009.07235). Per-function graph classification on CPG-style representations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser / S3 Frontend                                          │
│  React + Vite  →  http://graphault-frontend.s3-website...       │
│  Paste function → /explain → highlight risky AST nodes          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (port 80)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  EC2 t3.micro  (ap-south-1)                                     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ nginx                                                    │   │
│  │  • reverse proxy  :80 → localhost:8000                   │   │
│  │  • rate limit     10 req/min per IP on /predict /explain │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │ 127.0.0.1:8000 (loopback only)     │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │ Docker container: graphault-api                          │   │
│  │                                                          │   │
│  │  FastAPI (uvicorn)                                       │   │
│  │   POST /predict   → risk score + label                   │   │
│  │   POST /explain   → score + per-node saliency            │   │
│  │   GET  /model-info → metrics, threshold, approach        │   │
│  │   GET  /health    → {"status":"ok","model_loaded":true}  │   │
│  │                                                          │   │
│  │  CodeRiskGNN loaded once at startup (CPU inference)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  fail2ban  •  PasswordAuthentication no  •  key-pair SSH only   │
└─────────────────────────────────────────────────────────────────┘
                            │
                      Training data
                            │
              ┌─────────────▼──────────────┐
              │  MongoDB Atlas             │
              │  labeled_functions         │
              │  ~20k samples              │
              │  CVEfixes + BugsinPy       │
              │  + GitHub OSS              │
              └────────────────────────────┘
```

---

## Metrics

Evaluated on a held-out validation set with zero function-name overlap with training.

| Metric | Value |
|---|---|
| Validation PR-AUC | **0.2414** |
| Random baseline PR-AUC | 0.075 |
| Uplift over random | **~3.2x** |
| Validation F1 | 0.27 (at threshold 0.5417) |
| Class imbalance (pos weight) | 12.32:1 |
| Training samples | ~20,000 functions |

**On the numbers:** PR-AUC of 0.24 sounds low — it is, and it's honest. The dataset is severely imbalanced (1 vulnerable function per ~12 clean), the node features are shallow (type one-hots, no token semantics), and the task is genuinely hard. The 3.2x uplift over a random classifier means the model is learning real structural signal. The planned CodeBERT upgrade (see Roadmap) is where the precision jump will come from.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, PyTorch Geometric, GCNConv (3-layer) |
| Graph builder | Python `ast` module, custom CPG-style edges |
| Training data | MongoDB Atlas — CVEfixes, BugsinPy, GitHub OSS |
| API | FastAPI, uvicorn, Pydantic |
| Frontend | React 18, Vite 5 |
| Containerisation | Docker |
| Serving | nginx (reverse proxy + rate limiting) |
| Deployment | AWS EC2 t3.micro (API), AWS S3 static (frontend) |
| Security | fail2ban, key-pair SSH only, loopback port binding |

---

## API Reference

Base URL: `http://43.205.146.154` — interactive docs at `/docs`

### `POST /predict`

```bash
curl -X POST http://43.205.146.154/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "def get_user(name):\n    query = \"SELECT * FROM users WHERE name = \" + name\n    return db.execute(query)"}'
```

```json
{
  "risk_score": 0.1868,
  "label": 0,
  "num_nodes": 20
}
```

### `POST /explain`

Same request body. Returns the score plus per-node gradient saliency (top 10 nodes):

```json
{
  "risk_score": 0.1868,
  "label": 0,
  "top_nodes": [
    { "node_index": 6, "node_type": "arg",    "lineno": 1, "contribution": 1.0  },
    { "node_index": 3, "node_type": "Assign", "lineno": 2, "contribution": 0.84 },
    { "node_index": 8, "node_type": "BinOp",  "lineno": 2, "contribution": 0.62 }
  ]
}
```

### `GET /model-info`

Returns model architecture metadata, val metrics, threshold, and dataset description.

---

## Local Setup

**Prerequisites:** Python 3.11+, Node 18+, Docker (optional)

### 1. Clone and install

```bash
git clone https://github.com/jayeshmehra344/codesense.git
cd codesense
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-api.txt
```

### 2. Environment

```bash
# Create .env in the project root:
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/...
MONGO_DB_NAME=codesense
GITHUB_TOKEN=<your_pat>
```

### 3. Run the API

```bash
uvicorn src.api.app:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 4. Run the frontend

```bash
cd src/frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

### 5. Train (requires MongoDB with labeled data)

```bash
python src/model/train.py           # 50 epochs, saves data/model.pt
python src/model/find_threshold.py  # PR curve + F1-optimal threshold
```

### 6. Docker

```bash
docker build -t graphault-api .
docker run -p 127.0.0.1:8000:8000 graphault-api
```

---

## Project Structure

```
codesense/
├── src/
│   ├── api/
│   │   └── app.py                # FastAPI service (predict, explain, model-info)
│   ├── model/
│   │   ├── gnn.py                # CodeRiskGNN — 3-layer GCN
│   │   ├── train.py              # training loop, MongoDB data loader
│   │   ├── dataset.py            # PyG Dataset wrapper
│   │   └── find_threshold.py     # PR curve + F1-optimal threshold finder
│   ├── parser/
│   │   └── ast_graph_builder.py  # code → PyG Data (nodes, edges, features)
│   ├── graph/
│   │   ├── db.py                 # MongoDB connection
│   │   ├── labeler.py            # vulnerability label assignment
│   │   └── pipeline.py           # end-to-end data pipeline
│   ├── data/
│   │   ├── cvefixes_loader.py    # CVEfixes dataset ingestion
│   │   ├── bugsinpy_loader.py    # BugsinPy dataset ingestion
│   │   └── github_loader.py      # GitHub OSS clean-function sampling
│   └── frontend/
│       └── src/App.jsx           # single-file React dashboard
├── data/
│   └── model.pt                  # trained weights (gitignored — 309 MB)
├── Dockerfile
├── docker-compose.yml
├── requirements-api.txt
└── .github/
    └── workflows/
        └── graphault.yml         # PR analysis GitHub Action
```

---

## Roadmap

**Higher-quality features**
- [ ] Replace 89-dim node-type one-hot with 768-dim **CodeBERT** embeddings — only `ast_graph_builder.py` and the model's input layer change; API contract stays identical
- [ ] Add data-flow edges (def→use chains) to complement the current structural + control-flow edges
- [ ] Experiment with **GraphSAGE** or **GAT** in place of GCN for better neighbourhood aggregation on larger functions

**Infrastructure**
- [ ] HTTPS via Let's Encrypt / ACM (currently HTTP only)
- [ ] CI/CD: GitHub Action → rebuild and push Docker image on merge to `master`
- [ ] Structured logging + CloudWatch metrics for latency and error rate
- [ ] Domain name in place of bare IP

**Model**
- [ ] Scale training corpus from ~20k to 200k+ functions with broader CVE coverage
- [ ] Multi-language support via tree-sitter (JavaScript, TypeScript, C)
- [ ] Human-in-the-loop retraining: flagged predictions → review → retrain pipeline

---

## Why the numbers look like this

The model is intentionally honest about its current limits:

- **Shallow features** — node-type one-hots carry structure but no semantics. A `Call` node looks identical whether it calls `eval()` or `len()`.
- **Label noise** — CVE-linked commits often touch multiple functions; adjacent non-vulnerable functions get mislabelled positive.
- **Class imbalance** — 12:1 ratio structurally caps precision at low recall thresholds regardless of model quality.

The 3.2x uplift over random is real and reproducible. CodeBERT node features are the single highest-leverage next step to closing the gap with SOTA vulnerability detectors.
