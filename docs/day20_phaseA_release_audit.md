# Day 20 · Phase A — Release Audit & Safe Cleanup

*Goal: before writing any release doc, find out what is actually true about the repo, and make only
behaviour-preserving cleanups. Every later README / resume / interview claim must trace to this ledger.*

---

## 1. What the audit found

| Finding | Action |
|---|---|
| `README.md` is **0 bytes** | Author in Phase B |
| `LICENSE` was **0 bytes** | ✅ MIT, copyright holder `yashraj-a11y` (committed) |
| No `CHANGELOG.md`, `CONTRIBUTING.md`, lint config | CHANGELOG/CONTRIBUTING in Phase B; ✅ `ruff.toml` added |
| `docs/project_architecture.md` is a stale Day-1 layout (lists `06_XGBoost`) | Replace with `SYSTEM_ARCHITECTURE.md` in Phase C |
| 22 lint errors (pyflakes): 17 unused imports, 2 empty f-strings, 3 unused variables | ✅ 19 fixed; 3 unused variables left (see §3) |
| No `__pycache__` tracked, no `TODO/FIXME`, no `print()` in `src/` (168 modules) | Nothing to do |
| Only two tracked files >5 MB (`artifacts/fused_features/*.npz`, ~14 MB each) | Kept (small enough for GitHub, needed by tests) |
| Test runs rewrite `artifacts/testing/*.json` | Restored after each audit run; commit only when intentional |

## 2. Behaviour-preserving proof

| | Tests | Time |
|---|---|---|
| **Before** cleanup | **95 passed** | 5m27s |
| **After** cleanup | **95 passed** | 5m31s |

Also checked after the edit: `compileall` clean, and **every module under `src/` still imports**
(0 import failures). Lint rules are deliberately narrow (`E9`, `F` — real errors only); the default
ruff ruleset flagged 688 style items, which would be churn, not cleanup.

## 3. Deliberately NOT changed

- **3 unused local variables** (`bench` in `models/evaluation.py`, `sig_raw` in
  `pattern_mining/pattern_ranker.py`, `vi` in `reporting/investment_memo.py`): removing them is an
  "unsafe" fix (the right-hand side may have side effects). Left as-is.
- **Notebooks 01–19**: executed verification records, kept. Day 14 has no notebook (its work is `src/api/`).
- **Style-level lint (688 items)**: not release-relevant, and each edit risks a regression for no reader benefit.

## 4. Evidence ledger — what we may claim

✅ = backed by code/artifact (cite the path) · ⚠️ = claim only in the stated words · ❌ = do **not** claim

| Claim | Verdict | Evidence |
|---|---|---|
| Text normalization + TF-IDF | ✅ **800 features** (not 2,500), ngram (1,2) | `src/features/tfidf_vectorizer.py` |
| StandardScaler / OneHotEncoder | ✅ | `src/preprocessing/scaling.py`, `encoding.py` |
| Sparse–dense fusion (`scipy.sparse.hstack`) | ✅ | `src/features/fusion.py` |
| Logistic Regression baseline, coefficient analysis | ✅ | `src/training/`, `src/explainability/coefficient_analysis.py` |
| XGBoost, LightGBM, HistGB, RandomForest, ExtraTrees compared | ✅ | `docs/day4_benchmark_report.md`, `artifacts/evaluations/benchmark.json` |
| Deployed model = tuned LightGBM + sigmoid calibration, threshold 0.329 | ✅ | `artifacts/models/model_metadata.json` (val ROC-AUC 0.7767, PR-AUC 0.6884, Brier 0.1872) |
| PR / ROC / confusion / calibration plots | ✅ | `artifacts/evaluations/*.png` |
| SHAP explainability (+ additivity guard) | ✅ | `src/explainability/shap_explainer.py` |
| SBERT embeddings + FAISS semantic retrieval | ✅ | `src/retrieval/embedding_generator.py` (`all-MiniLM-L6-v2`) |
| Pattern mining | ✅ association rules (support/confidence/lift) + significance + clustering — **not** "Apriori" by name | `src/pattern_mining/` |
| Counterfactual, objection simulator, evolution planner, scenario simulation | ✅ | `src/counterfactual/`, `src/simulation/`, `src/evolution/`, `src/scenario/` |
| Recommendation engine, multi-agent (6 agents: 5 specialists + investment, plus consensus) | ✅ | `src/recommendation/`, `src/agents/` |
| Founder / investor reports | ✅ | `src/reporting/`, `outputs/reports/` |
| FastAPI backend, React frontend | ✅ | `src/api/`, `frontend/` |
| MLflow experiment tracking | ✅ local backend | `src/training/train_pipeline.py`, `src/mlops/` |
| 95 automated tests | ✅ all pass locally | this audit |
| CI | ⚠️ **fast lane only** (heavy tests need git-ignored artifacts) | `.github/workflows/ci.yml` |
| Dockerized | ⚠️ **Dockerfile/Compose authored, never built** (no Docker on dev machine) | `Dockerfile`, `docker-compose.yml` |
| Monitoring | ⚠️ health/live/ready probes + structured logging — not a metrics stack | `src/api/routes/health.py`, `src/utils/logger.py` |
| Market timing | ⚠️ **category-trend signal from Kickstarter data**; TAM/SAM/CAGR deferred | `src/copilot/market_analysis.py` |
| UI works end-to-end | ✅ rendered and driven in headless Chrome (Founder + Investor flows, mobile 320–390 px, no console errors) — ⚠️ not hand-tested by a person | `docs/assets/screenshots/`, `scripts/capture_screenshots.mjs` (Phase G) |
| Neural Network | ❌ never built | — |
| TF-IDF 2,500 | ❌ it is 800 | — |
| Founder-team-gap analysis, vision-language, RL | ❌ no code | — |
| Business-Model / Founder-Strength / Scalability / Market-Fit / Growth-Potential scores, and market size (TAM/SAM) | ❌ deferred by design (no data basis) — *corrected by the final verification notebook: earlier drafts listed "Market-Opportunity" as the sixth, but a category-level opportunity label IS computed (`market_opportunity` in `src/copilot/market_analysis.py`); only market size is deferred* | `VentureScoreGenerator.deferred` (5 scores, `value=None`); `market_size_tam` in the deferred appendix |
| PDF endpoint, real authentication/JWT | ❌ client print-to-PDF; auth-ready stubs only | `frontend/README.md` |

## 5. Interview soundbite

*"Before I wrote a line of release documentation I audited the repo: I found the README and license
were literally empty, built an evidence ledger mapping every claim I wanted to make to a file or
artifact, and dropped the ones I couldn't prove — a neural network, a 2,500-feature TF-IDF, founder-gap
scoring. The cleanup was behaviour-preserving by construction: I ran all 95 tests before and after,
checked every module still imported, and left the changes that could alter runtime behaviour alone."*
