# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/);
dates are commit dates.

## [1.0.0] — 2026-11-03 · First complete release

The platform is end to end: data → model → intelligence layers → API → dashboard.

### Fixed — found by rendering the UI (Day 20, Phase G)
- Investor portal showed **"Do Not Invest" in a green badge** (the colour test matched "invest"); verdicts
  now map to red / amber / green correctly.
- Mobile layout overflowed (header pushed the page past a 390 px screen); the header now wraps and both
  portal grids are constrained. Verified at 390, 360 and 320 px.
- Launch / deadline fields stack so the time is no longer clipped.
- Removed internal "Day-12 / Day-11" wording from the verdict note.

### Added — release (Day 20)
- `notebooks/20_Final_Release.ipynb`: re-verifies the release (links, quoted metrics vs artifacts, API contract vs
  published schema, all 95 tests, frontend build, assets, repo hygiene) — 78 checks pass, 0 fail.
- MIT `LICENSE`, root `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, project / install / deployment docs.
- Release audit and evidence ledger (`docs/day20_phaseA_release_audit.md`); conservative `ruff.toml`.
- Architecture, API, performance, demo, interview and portfolio/resume documents, plus a research roadmap
  (all proposals labelled not built); portfolio assets (`docs/assets/`): 19 UI screenshots,
  11 rendered diagrams, evaluation and SHAP charts, sample reports, example API responses, social preview card.

### Added — frontend (Day 19, 2026-11-02)
- React + TypeScript + Vite + Tailwind Founder and Investor portals with an auth-ready (stub) login.
- Prediction, SHAP bar chart and **waterfall**, retrieval, scenario simulator with counterfactuals,
  report viewer with provenance badges and print-to-PDF; deferred scores shown as visible cards.
- `/explain` now also returns `base_value` and `margin` (additive fields).
- Prediction tile now says *Above / Below the decision threshold* with a caption and a three-zone colour
  (red below threshold, amber between threshold and 50%, green above 50%), instead of "Funded" at a 40% probability.

### Added — deployment (Day 18, 2026-11-02)
- Multi-stage `Dockerfile`, Compose stack, nginx option, build / run / deploy / stop scripts.
- `/live` and `/ready` probes, expanded `/health`, structured logging, `.env` configuration, CORS
  allow-list (deny by default), deployment guide, cloud blueprint, security checklist.

### Added — testing & QA (Day 17, 2026-08-13)
- Model, retrieval, explainability, end-to-end and benchmark validation; API response-time budgets;
  CI fast lane (`pytest -m "not heavy"`).

### Added — MLOps (Day 16, 2026-08-11)
- MLflow experiment tracking, model registry, versioned artifacts, reproducible training pipeline.

### Added — retrieval evaluation (Day 15, 2026-08-05)
- Honest proxy metrics and latency benchmark for semantic retrieval.

### Added — backend (Day 14, 2026-08-05)
- FastAPI service over all layers: `/predict`, `/explain`, `/retrieve`, `/counterfactual`, `/scenario`,
  `/recommend`, `/analyze`, `/report/{founder,investor}`, `/health`.

### Added — intelligence layers (Days 6–13)
- Layer 13 reports (2026-08-05) · Layer 12 multi-agent system (2026-08-04) · Layer 11 recommendation
  engine (2026-08-04) · Layers 9–10 evolution planner and scenario simulator (2026-08-03) · Layer 8
  investor objection simulator and copilot (2026-08-03) · Layer 7 counterfactual engine (2026-07-31) ·
  Layer 6 pattern mining (2026-07-30) · Layers 2–3 knowledge base and semantic retrieval (2026-07-29).

### Added — modeling (Days 3–5)
- Explainability engine, SHAP (2026-07-28) · Prediction engine: Logistic Regression baseline, benchmark
  of LightGBM / XGBoost / HistGB / RandomForest / ExtraTrees, tuning, calibration (2026-07-27) ·
  Multi-modal feature engineering, TF-IDF (800) + scaled/encoded tabular fusion (2026-07-25).

### Added — foundations (Days 1–2)
- Production preprocessing pipeline (2026-07-24) · project setup, EDA, data-quality assessment (2026-07-23).

### Known limitations
- Business-Model, Founder-Strength, Scalability, Market-Fit and Growth-Potential scores, and market size
  (TAM / SAM / CAGR), are **deferred** (no data basis). Neural Network not built. Docker image not build-verified. No real
  authentication. CI runs the fast test lane only.
