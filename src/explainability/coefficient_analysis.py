"""Phase C — Logistic Regression Coefficient Analysis (Day-5, mandatory Week 7).

The white-box half of explainability. Logistic regression models the log-odds of
funding as a **linear** function of the 865 features:

    logit(p) = β₀ + Σ βᵢ xᵢ

so each coefficient βᵢ *is* the explanation — no post-hoc tool needed. Sign and
magnitude read directly:

- **βᵢ > 0** → feature pushes funding probability **up**; **βᵢ < 0** → pushes it down.
- Because the features were **StandardScaled** in Day 3, every βᵢ is on the same
  footing: it is the change in log-odds per **one standard deviation** of that feature.
  That is what makes |βᵢ| a legitimate cross-feature importance ranking (raw,
  differently-scaled coefficients would not be comparable).
- **exp(βᵢ)** is the odds-ratio: how the odds of funding multiply per +1 SD of the
  feature.

This module extracts all 865 coefficients, splits them into title TF-IDF tokens vs
tabular features (by NAME membership — reusing :func:`is_tabular` so it is robust to
the dropped-``goal`` index shift), and ranks the strongest positive / negative
drivers.

**Cross-model agreement (the key improvement).** An explanation is far more
trustworthy when *two independent models* agree. We merge the LightGBM mean|SHAP|
ranking from Phase B and report (a) the Spearman rank correlation between |β| and
mean|SHAP|, and (b) the features that land in the top-N of **both**. A feature both a
linear model and a gradient-boosted tree rank highly is strong evidence it is a real
driver, not an artifact of one model family.

Example
-------
>>> analyzer = CoefficientAnalyzer().load()
>>> analyzer.compute()               # DataFrame, 865 rows, ranked by |coef|
>>> analyzer.agreement()["spearman"] # rank correlation vs Phase-B SHAP
>>> analyzer.save()                  # -> coefficients.csv + report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.explainability.global_importance import is_tabular
from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class CoefficientResult:
    """Ranked coefficients + text/tabular partition + cross-model agreement."""

    coefficients: pd.DataFrame       # 865 rows: feature, group, coef, abs_coef, odds_ratio, direction, rank
    intercept: float
    n_text: int
    n_tabular: int
    agreement: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class CoefficientAnalyzer:
    """Extract, rank and cross-validate logistic-regression coefficients.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.explain.coefficient_analysis")

        c = self.config.get("day5", {}).get("coefficient_analysis", {})
        self.model_path = Path(c.get("model_path", "artifacts/models/logistic_regression.joblib"))
        self.feature_meta_path = Path(
            c.get("feature_meta_path", "artifacts/feature_pipeline/feature_metadata.json")
        )
        self.global_importance_csv = Path(
            c.get("global_importance_csv", "artifacts/explainability/global_importance.csv")
        )
        self.top_k = int(c.get("top_k", 25))
        self.agreement_top_n = int(c.get("agreement_top_n", 20))
        self.out_dir = Path(c.get("out_dir", "artifacts/explainability"))
        self.csv_path = Path(c.get("csv_path", self.out_dir / "coefficients.csv"))
        self.report_path = Path(c.get("report_path", "docs/day5_coefficient_report.md"))

        self.model_ = None
        self.feature_names_: list[str] = []
        self.result_: Optional[CoefficientResult] = None

    # ------------------------------------------------------------------ public
    def load(self) -> "CoefficientAnalyzer":
        """Load the L2 logistic-regression model and the 865 feature names."""
        self.model_ = joblib.load(self.model_path)
        meta = json.loads(self.feature_meta_path.read_text())
        self.feature_names_ = meta["feature_names"]
        n = len(self.feature_names_)
        if n != self.model_.coef_.shape[1]:
            raise ValueError(
                f"Name/model mismatch: {n} names vs {self.model_.coef_.shape[1]} coefficients."
            )
        self.logger.info("Loaded LogReg (%d coefficients) + %d names.",
                         self.model_.coef_.shape[1], n)
        return self

    def compute(self) -> pd.DataFrame:
        """Build the ranked coefficient table (by |coef|), with odds ratios."""
        if self.model_ is None:
            self.load()

        names = self.feature_names_
        coef = self.model_.coef_.ravel()
        df = pd.DataFrame({
            "feature": names,
            "group": ["tabular" if is_tabular(x) else "text" for x in names],
            "coef": coef,
            "abs_coef": np.abs(coef),
            "odds_ratio": np.exp(coef),               # odds multiplier per +1 SD
            "direction": np.where(coef >= 0, "up", "down"),
        })
        df["rank"] = df["abs_coef"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

        n_text = int((df["group"] == "text").sum())
        n_tab = int((df["group"] == "tabular").sum())
        agreement = self._agreement(df)

        self.result_ = CoefficientResult(
            coefficients=df,
            intercept=float(self.model_.intercept_[0]),
            n_text=n_text, n_tabular=n_tab,
            agreement=agreement,
            meta={"model": "LogisticRegression(L2)", "n_features": len(names)},
        )
        self.logger.info(
            "Ranked %d coefficients (%d text / %d tabular). Top driver: %s (%+.3f). "
            "Cross-model Spearman vs SHAP: %s.",
            len(df), n_text, n_tab, df.iloc[0]["feature"], df.iloc[0]["coef"],
            agreement.get("spearman"),
        )
        return df

    def top_positive(self, k: Optional[int] = None) -> pd.DataFrame:
        """Features that most increase funding probability (largest +coef)."""
        df = self._ensure().coefficients
        return df[df["coef"] > 0].sort_values("coef", ascending=False).head(k or self.top_k)

    def top_negative(self, k: Optional[int] = None) -> pd.DataFrame:
        """Features that most decrease funding probability (most negative coef)."""
        df = self._ensure().coefficients
        return df[df["coef"] < 0].sort_values("coef").head(k or self.top_k)

    def agreement(self) -> dict[str, Any]:
        """Cross-model agreement summary vs the Phase-B SHAP ranking."""
        return self._ensure().agreement

    def save(self) -> dict[str, Path]:
        """Persist the full coefficient CSV and a Markdown report."""
        self._ensure()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_.coefficients.to_csv(self.csv_path, index=False)
        self.report_path.write_text(self._render_report())
        self.logger.info("Saved coefficients -> %s and report -> %s",
                         self.csv_path, self.report_path)
        return {"csv": self.csv_path, "report": self.report_path}

    # --------------------------------------------------------------- internals
    def _ensure(self) -> CoefficientResult:
        if self.result_ is None:
            self.compute()
        return self.result_

    def _agreement(self, coef_df: pd.DataFrame) -> dict[str, Any]:
        """Rank-correlate |coef| with Phase-B mean|SHAP|; list top-N overlap.

        Both are *unsigned* importance rankings, so Spearman on |coef| vs
        mean|SHAP| measures whether the two model families agree on *which* features
        matter (not on direction). Skips gracefully if Phase B hasn't run.
        """
        if not self.global_importance_csv.exists():
            self.logger.warning("Global-importance CSV missing (%s) — skipping agreement.",
                                self.global_importance_csv)
            return {"available": False}

        shap_df = pd.read_csv(self.global_importance_csv)[["feature", "mean_abs_shap"]]
        merged = coef_df[["feature", "group", "abs_coef", "coef"]].merge(
            shap_df, on="feature", how="inner"
        )
        # Spearman = Pearson on the ranks; avoids a scipy dependency.
        r_coef = merged["abs_coef"].rank()
        r_shap = merged["mean_abs_shap"].rank()
        spearman = float(np.corrcoef(r_coef, r_shap)[0, 1])

        n = self.agreement_top_n
        top_coef = set(coef_df.head(n)["feature"])
        top_shap = set(shap_df.sort_values("mean_abs_shap", ascending=False).head(n)["feature"])
        overlap = sorted(top_coef & top_shap)

        return {
            "available": True,
            "spearman": round(spearman, 4),
            "top_n": n,
            "overlap_count": len(overlap),
            "overlap_features": overlap,
        }

    def _render_report(self) -> str:
        r = self.result_
        ag = r.agreement

        def table(sub: pd.DataFrame) -> str:
            head = "| feature | group | coef | odds_ratio (per +1 SD) |\n|---|---|---:|---:|\n"
            rows = [
                f"| `{row['feature']}` | {row['group']} | {row['coef']:+.3f} | {row['odds_ratio']:.3f} |"
                for _, row in sub.iterrows()
            ]
            return head + "\n".join(rows)

        if ag.get("available"):
            overlap = ", ".join(f"`{f}`" for f in ag["overlap_features"]) or "—"
            agreement_md = (
                f"- **Spearman(|coef|, mean\\|SHAP\\|) = {ag['spearman']:.3f}** over all "
                f"{r.meta['n_features']} features.\n"
                f"- **{ag['overlap_count']} / {ag['top_n']}** features appear in the top-"
                f"{ag['top_n']} of *both* rankings: {overlap}.\n"
            )
        else:
            agreement_md = "- Phase-B SHAP ranking not found — run Phase B for cross-model agreement.\n"

        return f"""# Day 5 · Phase C — Logistic Regression Coefficient Analysis (Week 7)

> Model: **{r.meta['model']}**, intercept β₀ = {r.intercept:+.4f}.
> {r.meta['n_features']} coefficients = **{r.n_text} title TF-IDF tokens + {r.n_tabular} tabular**.
> Features are StandardScaled ⇒ each coef = Δlog-odds per **+1 SD**; exp(coef) = odds multiplier.

## What this answers
The **white-box** view: the logistic model's coefficient *is* the explanation.
Positive → pushes funding probability up; negative → down. Ranked by |coef| because,
on standardized features, magnitude is directly comparable across features.

## Top {self.top_k} features that INCREASE funding probability (+coef)
{table(self.top_positive())}

## Top {self.top_k} features that DECREASE funding probability (−coef)
{table(self.top_negative())}

## Cross-model agreement (vs Phase-B LightGBM SHAP)
An independent tree model and this linear model agreeing on which features matter is
stronger evidence than either alone:

{agreement_md}
## Honesty caption
Coefficients are **associations learned from campaign metadata and title text**, not
causal levers. A negative coefficient on `youtube`/`website`/`app` means campaigns whose
titles lean on external-link words *tended* to fail in the training data — not that
removing the word causes funding. Only real features appear here; the dataset has no
founder/team/market/revenue signals.
"""
