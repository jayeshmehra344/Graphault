# codesense — Dev Log

---
2026-05-27
---

**Files modified today:**
- `src/data/github_loader.py` (15:25)
- `src/data/combine.py` (10:25)
- `data/training_data.json` (10:25)
- `data/combine.py` (09:58)
- `src/data/bugsinpy_loader.py` (09:24)

**Summary:**
Active day on the data pipeline — multiple loader scripts touched (GitHub, BugsinPy) and training data updated. The `combine.py` in both `src/data/` and `data/` were modified in the morning, suggesting work on merging/normalising data from multiple sources into a unified training set.

**Project structure snapshot:**
```
codesense/
├── src/
│   ├── api/          # empty — not yet built
│   ├── frontend/     # empty — not yet built
│   ├── data/
│   │   ├── github_loader.py      # fetches repos via GitHub API, stores in MongoDB
│   │   ├── bugsinpy_loader.py    # loads BugsinPy bug dataset
│   │   ├── cvefixes_loader.py    # loads CVEFixes vulnerability dataset
│   │   └── combine.py            # merges all data sources
│   ├── graph/
│   │   ├── pipeline.py           # clones repo → parses → saves graph to DB
│   │   ├── labeler.py            # labels graph nodes (buggy/clean)
│   │   └── db.py                 # MongoDB interface
│   ├── parser/
│   │   ├── parse.py              # AST parser — extracts functions & features
│   │   └── visualize.py         # graph visualisation (matplotlib)
│   └── model/
│       ├── gnn.py                # 3-layer GCN (CodeRiskGNN) — PyTorch Geometric
│       └── dataset.py            # PyTorch dataset wrapper
├── data/
│   ├── training_data.json        # compiled training set
│   ├── graph.json                # serialised graph
│   ├── cloned/                   # cloned repos (e.g. flask) for analysis
│   └── combine.py
├── tmp/
│   ├── bugsinpy/                 # BugsinPy raw data
│   └── github_repos/             # temp cloned repos during pipeline runs
├── notebooks/                    # empty
├── venv/                         # Python virtual environment
├── requirements.txt              # pinned deps (torch, torch-geometric, pymongo, etc.)
└── README.md                     # empty
```

**Current stage:**
Data collection and preprocessing is in active development. Three data sources are being wired together (GitHub API, BugsinPy, CVEFixes). The GNN model architecture (3-layer GCN for code risk classification) is defined but not yet trained. The graph construction pipeline is functional — it clones repos, parses ASTs, and stores structured graphs in MongoDB. API and frontend layers are stubbed but empty.

**Stack:** Python · PyTorch · PyTorch Geometric · MongoDB · NetworkX · HuggingFace Datasets

---
2026-05-29
---

**Files modified today:**
- `github_loader_run.log` (11:51)
- `src/model/gnn.py` (12:06)
- `src/parser/ast_graph_builder.py` (12:41)
- `src/model/train.py` (12:48)
- `data/model.pt` (12:54)
- `src/model/find_threshold.py` (16:49)
- `data/pr_curve.png` (16:51)
- `src/api/app.py` (16:51)
- `src/frontend/.gitignore` (16:57)
- `src/frontend/README.md` (16:57)
- `src/frontend/eslint.config.js` (16:57)
- `src/frontend/public/favicon.svg` (16:57)
- `src/frontend/public/icons.svg` (16:57)
- `src/frontend/src/App.css` (16:57)
- `src/frontend/src/assets/hero.png` (16:57)
- `src/frontend/src/index.css` (16:57)
- `src/frontend/src/main.jsx` (16:57)
- `src/frontend/vite.config.js` (16:57)
- `src/frontend/src/App.jsx` (16:59)
- `src/frontend/index.html` (17:00)
- `src/frontend/package.json` (17:40)
- `src/frontend/package-lock.json` (17:47)
- `.github/workflows/graphault.yml` (18:07)
- `scripts/run_analysis.py` (18:08)
- `analysis_comment.md` (18:12)
- `scripts/test_action_local.py` (18:12)
- `Dockerfile` (18:33)
- `docker-compose.yml` (18:33)
- `.dockerignore` (18:33)
- `requirements-api.txt` (18:33)
- `.claude/settings.local.json` (18:36)

**Summary:**
Highly active day spanning the full stack — the morning session focused on the ML core: `gnn.py` and `ast_graph_builder.py` were updated, `train.py` was run to completion (producing `data/model.pt`), and `find_threshold.py` was created to tune the classification threshold with a PR curve. The afternoon saw a major expansion: the previously empty `src/api/app.py` was scaffolded, the entire `src/frontend/` (Vite + React) was initialised with components and assets, a GitHub Actions workflow (`graphault.yml`) and analysis scripts were added, and the project was containerised with a `Dockerfile` and `docker-compose.yml` — indicating a push toward a deployable, end-to-end application.

