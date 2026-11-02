"""Phase F — Feature Importance Visualizations (Day-5, mandatory Week 7).

Turns the numbers from Phases B–E into the plots humans actually read. Nothing new
is computed — every figure consumes an artifact already on disk:

- ``global_importance.csv`` (Phase B) → mean|SHAP| bars (all / tabular / title tokens).
- ``coefficients.csv`` (Phase C) → diverging top ± driver bar (white-box view).
- ``shap_values.pkl`` (Phase D) → SHAP **beeswarm** summary + one-campaign **waterfall**.

Two engineering choices:

- **Headless rendering.** ``matplotlib.use("Agg")`` before importing pyplot, so figures
  render to PNG with no display — safe in a script, notebook, or CI.
- **Bars & waterfall are drawn by hand; only the beeswarm uses shap.** Manual matplotlib
  gives full control and immunity to shap's plotting-API churn across versions; the
  beeswarm is the one plot shap does uniquely well (per-point, colour-by-value), so we
  let the library draw that — reloading the subsample's feature *values* (via the saved
  ``row_index``) so points can be coloured.

Example
-------
>>> viz = ExplainabilityVisualizer()
>>> paths = viz.generate_all()          # -> artifacts/visualizations/*.png
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.plotting import plt        # lazy headless pyplot ("Agg"): loaded on first plot

_POS = "#2a9d8f"   # green — increases funding probability
_NEG = "#e76f51"   # red   — decreases funding probability


class ExplainabilityVisualizer:
    """Render the Day-5 explainability figures from persisted artifacts.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.explain.visualization")

        c = self.config.get("day5", {}).get("visualization", {})
        self.out_dir = Path(c.get("out_dir", "artifacts/visualizations"))
        self.global_csv = Path(c.get("global_importance_csv",
                                      "artifacts/explainability/global_importance.csv"))
        self.coef_csv = Path(c.get("coefficients_csv",
                                    "artifacts/explainability/coefficients.csv"))
        self.shap_pkl = Path(c.get("shap_values_pkl",
                                    "artifacts/explainability/shap_values.pkl"))
        self.fused_dir = Path(c.get("fused_dir", "artifacts/fused_features"))
        self.top_n = int(c.get("top_n", 20))
        self.waterfall_row = int(c.get("waterfall_row", 0))
        self.dpi = int(c.get("dpi", 130))

    # ------------------------------------------------------------------ public
    def generate_all(self) -> dict[str, Path]:
        """Render every figure; return {name: path}. Missing inputs are skipped."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        if self.global_csv.exists():
            gi = pd.read_csv(self.global_csv)
            paths["shap_importance"] = self._bar_mean_shap(
                gi, None, "Top features by mean|SHAP| (all)", "shap_importance.png")
            paths["tabular_importance"] = self._bar_mean_shap(
                gi, "tabular", "Top tabular features by mean|SHAP|", "tabular_importance.png")
            paths["tfidf_importance"] = self._bar_mean_shap(
                gi, "text", "Top title tokens by mean|SHAP|", "tfidf_importance.png")
        else:
            self.logger.warning("Missing %s — skipping SHAP-importance bars.", self.global_csv)

        if self.coef_csv.exists():
            paths["coefficient_plot"] = self._coefficient_plot(pd.read_csv(self.coef_csv))
        else:
            self.logger.warning("Missing %s — skipping coefficient plot.", self.coef_csv)

        if self.shap_pkl.exists():
            payload = pickle.loads(self.shap_pkl.read_bytes())
            paths["shap_summary"] = self._shap_summary(payload)
            paths["shap_waterfall"] = self._shap_waterfall(payload)
        else:
            self.logger.warning("Missing %s — skipping SHAP summary/waterfall.", self.shap_pkl)

        self.logger.info("Rendered %d figures -> %s", len(paths), self.out_dir)
        return paths

    # --------------------------------------------------------------- bar plots
    def _bar_mean_shap(self, gi: pd.DataFrame, group: Optional[str],
                       title: str, fname: str) -> Path:
        df = gi if group is None else gi[gi["group"] == group]
        df = df.nlargest(self.top_n, "mean_abs_shap").iloc[::-1]   # largest at top
        fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(df))))
        ax.barh(df["feature"], df["mean_abs_shap"], color="#4c72b0")
        ax.set_xlabel("mean |SHAP| (log-odds)")
        ax.set_title(title)
        ax.margins(y=0.01)
        return self._save(fig, fname)

    def _coefficient_plot(self, coef: pd.DataFrame) -> Path:
        top_pos = coef[coef["coef"] > 0].nlargest(self.top_n // 2, "coef")
        top_neg = coef[coef["coef"] < 0].nsmallest(self.top_n // 2, "coef")
        both = pd.concat([top_neg, top_pos]).sort_values("coef")
        colors = [_POS if v > 0 else _NEG for v in both["coef"]]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(both))))
        ax.barh(both["feature"], both["coef"], color=colors)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_xlabel("logistic coefficient  (+ raises / − lowers funding odds)")
        ax.set_title("Strongest funding drivers — LogReg coefficients")
        return self._save(fig, "coefficient_plot.png")

    # --------------------------------------------------------------- SHAP plots
    def _shap_summary(self, payload: dict) -> Path:
        """Beeswarm via shap, reloading the subsample's feature values for colour."""
        import shap
        values = payload["values"]
        names = payload["feature_names"]
        X = self._subsample_features(payload)             # feature values for colour
        fig = plt.figure()
        shap.summary_plot(values, features=X, feature_names=names,
                          max_display=self.top_n, show=False, plot_size=(9, 7))
        fig = plt.gcf()
        fig.suptitle("SHAP summary (beeswarm) — base LightGBM", y=1.02, fontsize=11)
        return self._save(fig, "shap_summary.png")

    def _shap_waterfall(self, payload: dict) -> Path:
        """Hand-drawn waterfall for one campaign: base → top ± features → margin."""
        values = payload["values"]
        base = float(payload["base_value"])
        names = payload["feature_names"]
        i = min(self.waterfall_row, values.shape[0] - 1)
        sv = values[i]
        margin = base + sv.sum()

        k = self.top_n // 2
        order = np.argsort(np.abs(sv))[::-1]
        top = order[:k]
        other = float(sv[order[k:]].sum())               # collapse the tail into one bar

        # Build the running waterfall from base upward.
        labels = ["base value"] + [names[j] for j in top] + ["other features", "final margin"]
        steps = [None] + [float(sv[j]) for j in top] + [other, None]
        running = base
        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(labels))))
        for row, (lab, step) in enumerate(zip(labels, steps)):
            if step is None:                             # anchor bars (base, final)
                val = base if lab == "base value" else margin
                ax.barh(row, val, color="#8d99ae")
                ax.text(val, row, f" {val:+.2f}", va="center", fontsize=8)
            else:
                color = _POS if step > 0 else _NEG
                ax.barh(row, step, left=running, color=color)
                ax.text(running + step, row, f" {step:+.2f}", va="center", fontsize=8)
                running += step
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(base, color="0.6", lw=0.8, ls="--")
        ax.set_xlabel("log-odds margin contribution")
        ax.set_title(f"SHAP waterfall — one campaign (row {i}), margin={margin:+.2f}")
        return self._save(fig, "shap_waterfall.png")

    # --------------------------------------------------------------- internals
    def _subsample_features(self, payload: dict) -> Optional[np.ndarray]:
        """Densify the exact subsample rows (via saved row_index) for beeswarm colour."""
        idx = payload.get("row_index")
        if idx is None:
            return None
        X = sp.load_npz(self.fused_dir / "X_train_selected.npz").tocsr()
        return X[np.asarray(idx)].toarray()

    def _save(self, fig, fname: str) -> Path:
        path = self.out_dir / fname
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved %s", path)
        return path
