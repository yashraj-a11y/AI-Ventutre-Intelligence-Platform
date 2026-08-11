"""Day 16 · Phase D — Hyperparameter search & model comparison (each config a tracked run).

"Which model / which config?" must be answered with *evidence*, not a hunch. This orchestrates the
**existing** Day-4 machinery — it does not reimplement training:

- **cross-family comparison** — `ModelBenchmark` trains the tree/boosting candidates
  (RandomForest, XGBoost, LightGBM) at a fixed config and scores them on validation;
- **mandatory baseline** — `LogisticRegressionTrainer` fits the shipped `logreg_l2` baseline every
  candidate must beat;
- **randomized search** — `HyperparameterTuner` (RandomizedSearchCV, PR-AUC, stratified CV) searches
  the incumbent family (LightGBM) for a better configuration.

Every candidate is logged as its own MLflow run via the Phase-C :class:`ExperimentTracker`, then all
are ranked by validation **PR-AUC** → the best configuration. Results mirror to
``artifacts/search/hyperparameter_search.json``.

**Honesty:** *grid* search is a trivial `RandomizedSearchCV → GridSearchCV` swap over the same space
(random search is used here as it reuses the Day-4 tuner); *Bayesian* optimization is **deferred** —
optuna is not a dependency. A **Neural Network** candidate is **deferred** (none trained; torch is
SBERT-only). **Never overwrites** the incumbent's `best_config.json` / `benchmark.json` — search
outputs go to ``artifacts/search/``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.models.benchmarking import ModelBenchmark
from src.models.data_split import ModelData, ModelDataSplitter
from src.models.logistic_regression import LogisticRegressionTrainer
from src.models.tuning import HyperparameterTuner
from src.mlops.experiment_tracker import ExperimentTracker
from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_METRIC = "pr_auc"                      # primary ranking metric (imbalanced-aware)
_CAPTION = ("Candidates compared on the SAME held-out validation split, ranked by PR-AUC. LogReg is "
            "the mandatory baseline; LightGBM the incumbent family. Grid = trivial swap over the same "
            "space; Bayesian/optuna and a Neural-Network candidate are DEFERRED (no dep / none trained).")


class HyperparameterSearch:
    """Compare candidate model families + randomized search, tracking each config as an MLflow run."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 tracker: Optional[ExperimentTracker] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.training.search")

        s = self.config.get("day16", {}).get("search", {})
        self.candidates = list(s.get("candidates", ["logreg", "random_forest", "xgboost", "lightgbm"]))
        self.search_family = s.get("search_family", "lightgbm")
        self.n_iter = int(s.get("n_iter", 8))
        self.cv = int(s.get("cv", 3))
        self.out_path = _REPO_ROOT / s.get("out_path", "artifacts/search/hyperparameter_search.json")
        self.tracker = tracker or ExperimentTracker(self.config, self.logger)

    # ------------------------------------------------------------------ candidates
    def _logreg_baseline(self, data: ModelData) -> dict[str, Any]:
        """Mandatory baseline via the EXISTING trainer, limited to the shipped `logreg_l2` config."""
        trainer = LogisticRegressionTrainer(self.config, self.logger)
        trainer.penalties = ["l2"]                  # just the shipped baseline (fast); no save()
        results = trainer.train_all(data.X_fit, data.y_fit, data.X_val, data.y_val)
        r = next(x for x in results if x.name == "logreg_l2")
        return {"name": "logreg_l2", "kind": "baseline", "model_type": "LogisticRegression(l2)",
                "params": {"penalty": "l2", "C": trainer.C, "solver": trainer.solver},
                "metrics": r.metrics, "train_time_s": round(float(r.train_time_s), 1)}

    def _tree_family_comparison(self, families: list[str], data: ModelData) -> list[dict[str, Any]]:
        """Fixed-config comparison of tree/boosting families via the EXISTING ModelBenchmark."""
        bench = ModelBenchmark(self.config, self.logger)
        bench.model_names = families                # override the Day-4 list; don't call save_json()
        rows: list[dict[str, Any]] = []
        for res in bench.run(data.X_fit, data.y_fit, data.X_val, data.y_val):
            rows.append({"name": res.name, "kind": "comparison", "model_type": res.name,
                         "params": dict(res.params), "metrics": res.metrics,
                         "train_time_s": round(float(res.train_time_s), 1)})
        return rows

    def _random_search(self, data: ModelData) -> Optional[dict[str, Any]]:
        """Randomized search over the incumbent family via the EXISTING tuner (redirected output)."""
        tuner = HyperparameterTuner(self.config, self.logger)
        tuner.model_name = self.search_family
        tuner.n_iter, tuner.cv = self.n_iter, self.cv
        tuner.best_config_path = str(_REPO_ROOT / f"artifacts/search/best_config_{self.search_family}.json")
        try:
            res = tuner.tune(data.X_fit, data.y_fit, data.X_val, data.y_val)   # writes to search/ only
        except ValueError as e:                     # tuner has no space for this family
            self.logger.warning("Random search skipped for %r: %s", self.search_family, e)
            return None
        return {"name": f"{self.search_family}_search", "kind": "search",
                "model_type": f"{self.search_family} (RandomizedSearchCV)",
                "params": res["best_params"], "metrics": res["val_metrics"],
                "best_cv_score": res["best_cv_score"], "train_time_s": res["search_time_s"]}

    # ------------------------------------------------------------------ public
    def run(self, data: Optional[ModelData] = None, track: bool = True) -> dict[str, Any]:
        """Compare all candidates + random search; track each; rank by validation PR-AUC."""
        data = data or ModelDataSplitter(self.config, self.logger).load()
        rows: list[dict[str, Any]] = []

        if "logreg" in self.candidates:
            rows.append(self._logreg_baseline(data))
        tree = [c for c in self.candidates if c != "logreg"]
        if tree:
            rows.extend(self._tree_family_comparison(tree, data))
        srow = self._random_search(data)
        if srow:
            rows.append(srow)

        rows.sort(key=lambda r: r["metrics"][_METRIC], reverse=True)

        if track:
            for r in rows:
                self.tracker.track(run_name=f"search-{r['name']}", model_type=r["model_type"],
                                   params=r["params"], metrics=r["metrics"],
                                   tags={"phase": "D", "kind": r["kind"]},
                                   extra={"train_time_s": r["train_time_s"]})

        baseline = next((r for r in rows if r["kind"] == "baseline"), None)
        report = {
            "primary_metric": _METRIC, "n_candidates": len(rows),
            "best": rows[0] if rows else None,
            "baseline": baseline,
            "beats_baseline": (bool(rows and baseline and rows[0]["metrics"][_METRIC]
                                    > baseline["metrics"][_METRIC]) if baseline else None),
            "ranked": rows, "honesty_note": _CAPTION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(report, indent=2, default=str))
        if rows:
            self.logger.info("Search: best %s PR-AUC=%.4f (baseline logreg_l2 %.4f); ranked %d; -> %s",
                             rows[0]["name"], rows[0]["metrics"][_METRIC],
                             baseline["metrics"][_METRIC] if baseline else float("nan"),
                             len(rows), self.out_path)
        return report
