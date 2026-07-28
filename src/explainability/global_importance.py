"""Phase B — Global Feature Importance (Day-5, Layer 5).

*Which real features drive funding success across the whole model?* This is the
**global** question (Phase A): one importance score per feature, averaged over the
data — as opposed to the **local**, per-campaign question (Phases D/E).

We rank the 865 selected features three independent ways on the base tuned
**LightGBM** (``best_model.joblib``) and compare them:

- **gain**  — total loss reduction a feature contributed across all its splits.
  *This is the honest tree-importance* (how *useful* a feature was), read from the
  booster. It is NOT the sklearn default.
- **split** — how *often* a feature was used for a split (``feature_importances_``
  defaults to this). A frequently-used feature can still be low-gain, so we keep
  both and let the contrast be informative.
- **mean|SHAP|** — average absolute SHAP contribution over a densified subsample
  (~2,000 rows). SHAP is model-agnostic and additive, so its ranking is the one we
  most trust; gain/split are cheap tree-internal proxies.

Three non-obvious correctness points baked in:

1. **gain, not split.** ``LGBMClassifier.feature_importances_`` is *split* count by
   default; gain must be pulled from ``booster_.feature_importance("gain")``.
2. **text vs tabular split is by NAME membership, not column index.** Feature
   selection dropped ``goal`` (1 correlated column), so a fixed "first 800 = text"
   index rule would be off-by-one for every tabular feature. We classify each of the
   865 names by pattern (one-hot prefixes / known numeric names / ``category_freq``)
   → everything else is a title TF-IDF token.
3. **interventional SHAP, guarded by an additivity assertion.** SHAP's *default*
   fast path for LightGBM (``tree_path_dependent``) is **numerically broken for this
   model** — it returns contributions in the thousands that violate additivity by
   ~1e6 (verified). We use ``feature_perturbation="interventional"`` with a small
   background sample (correct, uses a reference distribution instead of the tree's
   internal cover) and **assert** ``expected + Σshap ≈ margin`` before trusting the
   ranking. This is the sanity check that caught the bug.

Example
-------
>>> analyzer = GlobalImportanceAnalyzer().load()
>>> ranking = analyzer.compute()          # DataFrame, 865 rows, ranked
>>> analyzer.save()                        # -> global_importance.csv + report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.config import load_config
from src.utils.logger import get_logger

# Tabular feature identity (everything else in the 865 is a title TF-IDF token).
# `goal` is intentionally absent — selection dropped it as correlated.
_NUMERIC_TABULAR = {
    "usd_goal_real", "goal_per_day", "campaign_duration",
    "launch_year", "launch_month", "launch_weekday", "launch_quarter",
    "is_weekend_launch", "title_length", "word_count", "avg_word_length",
    "uppercase_ratio", "has_number",
}
_TABULAR_PREFIXES = ("category_", "main_category_", "currency_", "country_")


def is_tabular(name: str) -> bool:
    """True if a feature NAME is a tabular feature (not a title TF-IDF token).

    Classification is by name membership so it is robust to the index shift caused
    by selection dropping ``goal``. ``category_freq`` is tabular and is caught by the
    ``category_`` prefix.
    """
    return (
        name in _NUMERIC_TABULAR
        or name == "category_freq"
        or name.startswith(_TABULAR_PREFIXES)
    )


@dataclass
class GlobalImportanceResult:
    """Ranked importances + the text/tabular partition + provenance."""

    ranking: pd.DataFrame            # 865 rows: feature, group, gain, split, mean_abs_shap, *_rank
    n_text: int
    n_tabular: int
    shap_n_rows: int
    top_k: int
    meta: dict[str, Any] = field(default_factory=dict)


class GlobalImportanceAnalyzer:
    """Rank all 865 features by gain / split / mean|SHAP| on the base LightGBM.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.explain.global_importance")

        c = self.config.get("day5", {}).get("global_importance", {})
        self.fused_dir = Path(c.get("fused_dir", "artifacts/fused_features"))
        self.model_path = Path(c.get("model_path", "artifacts/models/best_model.joblib"))
        self.feature_meta_path = Path(
            c.get("feature_meta_path", "artifacts/feature_pipeline/feature_metadata.json")
        )
        self.shap_subsample = int(c.get("shap_subsample", 2000))
        self.shap_background = int(c.get("shap_background", 200))
        self.shap_additivity_tol = float(c.get("shap_additivity_tol", 1e-3))
        self.random_state = int(c.get("random_state", 42))
        self.top_k = int(c.get("top_k", 50))
        self.out_dir = Path(c.get("out_dir", "artifacts/explainability"))
        self.csv_path = Path(c.get("csv_path", self.out_dir / "global_importance.csv"))
        self.report_path = Path(c.get("report_path", "docs/day5_global_importance_report.md"))

        self.model_ = None
        self.feature_names_: list[str] = []
        self.result_: Optional[GlobalImportanceResult] = None

    # ------------------------------------------------------------------ public
    def load(self) -> "GlobalImportanceAnalyzer":
        """Load the base LightGBM model and the 865 selected feature names."""
        self.model_ = joblib.load(self.model_path)
        meta = json.loads(self.feature_meta_path.read_text())
        self.feature_names_ = meta["feature_names"]
        n = len(self.feature_names_)
        if n != self.model_.n_features_in_:
            raise ValueError(
                f"Name/model mismatch: {n} names vs {self.model_.n_features_in_} model features."
            )
        self.logger.info("Loaded %s (%d features) + %d names.",
                         type(self.model_).__name__, self.model_.n_features_in_, n)
        return self

    def compute(self) -> pd.DataFrame:
        """Compute gain, split and mean|SHAP| importances; return the ranked table."""
        if self.model_ is None:
            self.load()

        names = self.feature_names_
        gain = self._gain_importance()
        split = np.asarray(self.model_.feature_importances_, dtype=float)  # sklearn default = split
        shap_abs, shap_rows, shap_add_err = self._mean_abs_shap()

        df = pd.DataFrame({
            "feature": names,
            "group": ["tabular" if is_tabular(x) else "text" for x in names],
            "gain": gain,
            "split": split,
            "mean_abs_shap": shap_abs,
        })
        # Normalised shares make magnitudes comparable across the three methods.
        for col in ("gain", "split", "mean_abs_shap"):
            total = df[col].sum()
            df[f"{col}_share"] = df[col] / total if total > 0 else 0.0
            df[f"{col}_rank"] = df[col].rank(ascending=False, method="min").astype(int)

        # Primary ordering = SHAP (most trustworthy), gain as tiebreaker.
        df = df.sort_values(["mean_abs_shap", "gain"], ascending=False).reset_index(drop=True)

        n_text = int((df["group"] == "text").sum())
        n_tab = int((df["group"] == "tabular").sum())
        self.result_ = GlobalImportanceResult(
            ranking=df, n_text=n_text, n_tabular=n_tab,
            shap_n_rows=shap_rows, top_k=self.top_k,
            meta={
                "model": type(self.model_).__name__,
                "n_features": len(names),
                "shap_subsample_requested": self.shap_subsample,
                "shap_perturbation": "interventional",
                "shap_additivity_error": shap_add_err,
            },
        )
        self.logger.info(
            "Ranked %d features (%d text / %d tabular). Top by SHAP: %s.",
            len(df), n_text, n_tab, df.iloc[0]["feature"],
        )
        return df

    def top(self, k: Optional[int] = None, group: Optional[str] = None) -> pd.DataFrame:
        """Top-k features overall or within a group ('text' / 'tabular')."""
        if self.result_ is None:
            self.compute()
        k = k or self.top_k
        df = self.result_.ranking
        if group is not None:
            df = df[df["group"] == group]
        return df.head(k)

    def save(self) -> dict[str, Path]:
        """Persist the full ranking CSV and a Markdown report."""
        if self.result_ is None:
            self.compute()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.result_.ranking.to_csv(self.csv_path, index=False)
        self.report_path.write_text(self._render_report())
        self.logger.info("Saved ranking -> %s and report -> %s", self.csv_path, self.report_path)
        return {"csv": self.csv_path, "report": self.report_path}

    # --------------------------------------------------------------- internals
    def _gain_importance(self) -> np.ndarray:
        """Gain importance from the booster (NOT the split-count default)."""
        booster = self.model_.booster_
        return np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)

    def _mean_abs_shap(self) -> tuple[np.ndarray, int, float]:
        """Mean |SHAP| over a densified subsample, via INTERVENTIONAL TreeSHAP.

        SHAP wants dense input; we densify only ~`shap_subsample` explain rows plus a
        small background sample (never the full 66k x 865). We use interventional
        perturbation because LightGBM's default fast TreeSHAP is broken for this model
        (see module docstring), then **assert additivity** — expected + Σshap ≈ raw
        margin — as a correctness guard before returning.

        Returns (mean_abs_shap[865], n_explain_rows, max_additivity_error).
        """
        import shap  # local import: heavy, and only needed here

        X = sp.load_npz(self.fused_dir / "X_train_selected.npz").tocsr()
        rng = np.random.default_rng(self.random_state)
        n = min(self.shap_subsample, X.shape[0])
        n_bg = min(self.shap_background, X.shape[0])
        idx = rng.choice(X.shape[0], size=n, replace=False)
        bg_idx = rng.choice(X.shape[0], size=n_bg, replace=False)
        X_sub = X[idx].toarray()        # explain rows  (~2000 x 865 ≈ 13 MB)
        background = X[bg_idx].toarray()  # reference distribution for interventional

        explainer = shap.TreeExplainer(
            self.model_,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",          # log-odds margin (calibration is a squash on top)
        )
        sv = _positive_class_shap(explainer.shap_values(X_sub))  # (n, 865), success class

        # --- additivity guard: expected_value + Σshap must equal the raw margin ------
        ev = explainer.expected_value
        ev = float(ev[1]) if np.ndim(ev) > 0 and len(np.atleast_1d(ev)) > 1 else float(np.atleast_1d(ev)[0])
        margin = self.model_.predict(X_sub, raw_score=True)
        add_err = float(np.abs(ev + sv.sum(axis=1) - margin).max())
        if add_err > self.shap_additivity_tol:
            raise ValueError(
                f"SHAP additivity violated: max|expected+Σshap-margin|={add_err:.3e} "
                f"> tol {self.shap_additivity_tol:.1e}. Attributions are untrustworthy."
            )

        mean_abs = np.abs(sv).mean(axis=0)
        self.logger.info(
            "mean|SHAP| on %d rows (bg=%d), additivity err=%.2e (tol %.1e) OK.",
            n, n_bg, add_err, self.shap_additivity_tol,
        )
        return np.asarray(mean_abs, dtype=float), n, add_err

    def _render_report(self) -> str:
        r = self.result_
        df = r.ranking
        top = df.head(r.top_k)

        def table(sub: pd.DataFrame) -> str:
            head = "| rank | feature | group | gain_share | split_share | mean\\|SHAP\\| |\n"
            head += "|---:|---|---|---:|---:|---:|\n"
            rows = []
            for i, (_, row) in enumerate(sub.iterrows(), start=1):
                rows.append(
                    f"| {i} | `{row['feature']}` | {row['group']} | "
                    f"{row['gain_share']:.4f} | {row['split_share']:.4f} | "
                    f"{row['mean_abs_shap']:.4f} |"
                )
            return head + "\n".join(rows)

        tab_top = df[df["group"] == "tabular"].head(15)
        text_top = df[df["group"] == "text"].head(15)

        return f"""# Day 5 · Phase B — Global Feature Importance

> Model: **{r.meta['model']}** (base tuned LightGBM, uncalibrated, SHAP-able).
> {r.meta['n_features']} features = **{r.n_text} title TF-IDF tokens + {r.n_tabular} tabular**.
> mean\\|SHAP\\| over **{r.shap_n_rows}** densified rows via **interventional** TreeSHAP
> (the default fast path is numerically broken for this model).
> Additivity guard: max\\|expected+Σshap−margin\\| = **{r.meta['shap_additivity_error']:.2e}** ✅.

## What this answers
The **global** question from Phase A: which real features drive funding success across
the whole model? Three independent rankings — **gain** (loss reduction, the honest tree
importance), **split** (usage count), and **mean\\|SHAP\\|** (average absolute
contribution, model-agnostic + additive) — cross-checked against each other. Primary
ordering below is by mean\\|SHAP\\|, gain as tiebreak.

## Top {r.top_k} features (by mean\\|SHAP\\|)
{table(top)}

## Top 15 tabular features
{table(tab_top)}

## Top 15 title TF-IDF tokens
{table(text_top)}

## Honesty caption
These are **associations the model learned from campaign metadata and title text**, not
causal drivers. A high mean\\|SHAP\\| means "this feature moved the model's funding
probability a lot on average," not "changing it will cause funding." Tabular features
(goal size, category commonness, timing) and specific title words are all the model saw —
there are no founder/team/market/revenue signals in this dataset.
"""


def _positive_class_shap(sv: Any) -> np.ndarray:
    """Normalise SHAP output across shap/LightGBM API variants to (n_rows, n_features).

    TreeExplainer on a binary LGBMClassifier may return:
      - a single 2D array (n, d)                      -> use as-is;
      - a list [class0, class1] of 2D arrays          -> take class 1 (success);
      - a 3D array (n, d, n_classes)                  -> take [..., 1].
    """
    if isinstance(sv, list):
        return np.asarray(sv[1] if len(sv) > 1 else sv[0])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[..., 1]
    return sv
