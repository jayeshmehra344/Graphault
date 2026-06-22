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

---
2026-06-03
---

**Files modified today:**
- `src/frontend/.env.production` (07:27)
- `src/api/app.py` (09:56)
- `graphault-api.tar.gz` (10:23)
- `src/frontend/vite.config.js` (10:26)
- `src/frontend/dist/assets/index-DWdAUg6-.js` (19:33)
- `src/frontend/dist/assets/index-I6cGQ-pn.css` (19:33)
- `src/frontend/dist/index.html` (19:33)
- `src/frontend/src/App.jsx` (19:33)
- `README.md` (19:52)
- `src/data/dedup.py` (20:28)
- `data/splits/dedup_meta.json` (20:48)
- `data/splits/test_ids.json` (20:48)
- `data/splits/train_ids.json` (20:48)
- `.claude/settings.local.json` (20:54)
- `src/model/precompute_codebert.py` (20:54)
- `data/codebert_node_features.pt` (21:00)
- `data/precompute_codebert.log` (21:00)
- `data/precompute_stdout.log` (21:00)
- `src/model/train.py` (21:04)
- `data/model_deduped_89dim.pt` (21:23)

**Summary:**
A full-stack day weighted toward the ML pipeline and deployment. The morning continued API and frontend work — `src/api/app.py` was updated, `vite.config.js` and `.env.production` were configured, and a `graphault-api.tar.gz` build artifact was produced, with a production frontend bundle later built into `src/frontend/dist/`. The evening focused on data quality and model improvements: `src/data/dedup.py` generated deduplicated train/test splits (`data/splits/`), `src/model/precompute_codebert.py` computed CodeBERT node-feature embeddings (`data/codebert_node_features.pt`), and `train.py` was rerun to produce a new deduplicated 89-dimensional model checkpoint (`data/model_deduped_89dim.pt`) — suggesting a move to richer CodeBERT-based node features for the GNN.

---
2026-06-05
---

**Files modified today:**
- `src/model/precompute_codebert.py` (07:19)
- `data/codebert_node_features.pt` (07:46)
- `data/precompute_codebert.log` (07:46)
- `data/precompute_stdout.log` (07:46)

**Summary:**
Short focused session revisiting the CodeBERT node-feature precomputation pipeline — `src/model/precompute_codebert.py` was updated and re-executed, producing a fresh `data/codebert_node_features.pt` embedding file along with updated log files (`precompute_codebert.log`, `precompute_stdout.log`). This mirrors the pattern from the June 3rd evening session, suggesting iterative refinement of the CodeBERT embedding step — likely adjusting how node features are extracted or batched before GNN training.

---
2026-06-07
---

**Files modified today:**
- `data/codebert_node_features.tmp` (05:46)
- `.claude/settings.local.json` (07:12)
- `src/model/precompute_codebert.py` (07:19)
- `data/codebert_node_features.pt` (07:46)
- `data/precompute_codebert.log` (07:46)
- `data/precompute_stdout.log` (07:46)

**Summary:**
Another focused session on the CodeBERT node-feature precomputation step — `src/model/precompute_codebert.py` was modified and rerun, regenerating `data/codebert_node_features.pt` and its associated logs. The presence of `data/codebert_node_features.tmp` (written earlier in the morning before the script completed) suggests the precomputation script was iterated on — likely tuning embedding batch size, feature dimensionality, or which AST node types are included — before a clean final run produced the `.pt` output.

---
2026-06-08
---

**Files modified today:**
- `src/model/precompute_codebert.py` (12:13)
- `data/precompute_codebert.log` (12:32)
- `data/precompute_stdout.log` (12:32)

**Summary:**
Another iteration on the CodeBERT node-feature precomputation pipeline — `src/model/precompute_codebert.py` was updated and a new run kicked off, with logs showing progress through the full 41,140-function dataset at ~4.9% completion by the last recorded checkpoint. The log output (tracking batch progress, skip counts, and cache size) indicates the script is actively embedding AST node sequences via CodeBERT for the GNN's node-feature input; the repeated modifications to this file over recent days suggest ongoing tuning of batching logic or embedding strategy before retraining the model.

---
2026-06-09
---

**Files modified today:** None

**Summary:**
No changes today.

---
2026-06-10
---

**Files modified today:** None

**Summary:**
No changes today.

---
2026-06-12
---

**Files modified today:**
- `.claude/settings.local.json` (04:46)
- `data/precompute_codebert.log` (04:54)
- `data/precompute_stdout.log` (04:54)

**Summary:**
Very early morning activity — the CodeBERT precomputation logs (`data/precompute_codebert.log`, `data/precompute_stdout.log`) were updated at 04:54, indicating the precompute script was run again overnight or in the early hours, continuing the pattern of iterative CodeBERT embedding runs seen across recent sessions. The `.claude/settings.local.json` update at 04:46 suggests a brief configuration adjustment just before the run. No source files were modified, pointing to a re-run of the existing `precompute_codebert.py` script rather than active development.

---
2026-06-12 (update)
---

**Files modified today (later in the day):**
- `src/scan/repo_scan.py` (06:43)
- `src/api/app.py` (07:01)

**Summary:**
Development resumed later in the morning with changes to two source files. `src/scan/repo_scan.py` — a new or recently added module under `src/scan/` — was modified at 06:43, suggesting work on a repository scanning capability that likely coordinates with the existing graph/pipeline infrastructure. `src/api/app.py` was updated at 07:01, indicating continued iteration on the Flask API layer — possibly adding a new endpoint to expose scan results or connecting the scan module to the API surface.

---
2026-06-13
---

**Files modified today:**
- `src/scan/repo_scan.py` (06:43)
- `src/api/app.py` (07:01)
- `src/frontend/src/RepoScan.jsx` (16:46)
- `src/frontend/src/shared.jsx` (16:46)
- `src/frontend/dist/assets/index-lymiIOyz.js` (16:47)

**Summary:**
Full end-to-end day delivering the repo-scan feature across all three layers. `src/scan/repo_scan.py` implements the core scanner — it walks a local Python repo, parses each `.py` file via the existing AST pipeline, scores every function through the loaded CodeRiskGNN model, and returns a structured report with per-function risk scores and a flagged-count summary. `src/api/app.py` was updated to import `scan_repo` and expose it as a new `POST /scan-repo` endpoint, wiring the scanner into the FastAPI service with full Pydantic request/response models. The afternoon session completed the UI side: `src/frontend/src/RepoScan.jsx` adds a new React view with a path input, scan button, summary stat boxes, and a sortable function risk table — while `src/frontend/src/shared.jsx` was refactored into a shared theme/primitive module (constants, colour palette, `scoreColor`, and reusable UI components) consumed by both the new scan view and the existing single-function analyzer. The production bundle in `dist/assets/` was rebuilt to include these changes.

---
2026-06-17
---

**Files modified today:**
- `data/precompute_codebert.log` (07:18)
- `src/parser/ast_graph_builder.py` (07:36)
- `data/scan_cache.log` (07:54)
- `.claude/settings.local.json` (08:06)
- `data/train_codebert.log` (08:24)
- `src/model/train.py` (08:27)
- `data/model_codebert_sanity.pt` (08:50)
- `data/train_codebert_sanity.log` (08:50)

**Summary:**
A focused ML pipeline session centred on the AST-to-GNN path. `src/parser/ast_graph_builder.py` was updated first, followed by a CodeBERT precompute run and a rebuild of `data/scan_cache.log` — suggesting the AST change altered how graph nodes are constructed, requiring downstream caches to be refreshed. `src/model/train.py` was then modified and a targeted "sanity" training run was executed, producing `data/model_codebert_sanity.pt` and `data/train_codebert_sanity.log` — a lightweight checkpoint used to quickly validate the pipeline end-to-end after the AST builder changes before committing to a full training run.

---
2026-06-17 (update)
---

**Files modified later in the day:**
- `src/frontend/.env.development` (11:30)
- `data/vite_dev.log` (11:45)
- `data/vite_dev2.log` (11:47)
- `data/uvicorn.log` (11:54)
- `data/uvicorn2.log` (12:01)
- `data/uvicorn2_err.log` (12:01)
- `data/uvicorn3_err.log` (12:06)
- `data/vite_final.log` (12:12)
- `data/vite_clean.log` (12:24)
- `data/uvicorn3.log` (12:44)

**Summary:**
An afternoon session focused on getting the full-stack dev environment running. `src/frontend/.env.development` was modified — likely updating the API base URL or environment variables for local development — followed by a series of Vite dev server (`vite_dev.log`, `vite_dev2.log`, `vite_final.log`, `vite_clean.log`) and Uvicorn backend server (`uvicorn.log`, `uvicorn2.log`, `uvicorn3.log`, error logs) runs, suggesting iterative troubleshooting of the frontend↔API connection in dev mode. Multiple restarts of both servers indicate debugging of CORS settings, proxy configuration, or port conflicts before a working local dev setup was established.

---
2026-06-19
---

**Files modified today:** None

**Summary:**
No changes today.

---
2026-06-21
---

**Files modified today:**
- `.claude/settings.local.json` (10:52)
- `src/model/train.py` (10:55)
- `data/smoke_test.log` (10:56)

**Summary:**
Short focused session on model training — `src/model/train.py` was modified and executed, with `data/smoke_test.log` produced as output, suggesting a lightweight smoke-test training run was used to validate the pipeline (likely following recent changes to the AST builder or CodeBERT precompute step). The `.claude/settings.local.json` update just before the run is consistent with a brief configuration adjustment prior to kicking off the training script.

