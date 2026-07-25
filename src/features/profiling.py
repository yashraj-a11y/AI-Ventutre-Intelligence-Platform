"""Phase A — Exploratory Feature Analysis (Day-3 Stage-2).

Before engineering anything, :class:`FeatureProfiler` measures the **training**
distribution of every feature and emits a *candidate-transformation list* — the
evidence that justifies each later phase (log-transform, one-hot vs frequency
encoding, dropping a correlated column, TF-IDF sizing).

Why train-only? Deciding "log-transform ``goal``" or "frequency-encode
``category``" is itself a choice *learned from data*. Learn it from the test set
and you have leaked — the same discipline as fitting a scaler. So the profiler
consumes ``X_train`` from Phase 0. It also runs one **informational** robustness
check against the test set (categories unseen in train); this never changes a
fitted parameter, it only sizes the ``handle_unknown`` risk for Phase B.

Decision rules (config-tunable)
-------------------------------
- numeric with ``|skew| > skew_threshold``      -> recommend ``log1p``
- categorical with ``nunique <= onehot_max``    -> one-hot; else frequency-encode
- numeric pair with ``|corr| > corr_threshold`` -> recommend dropping one
- engineered ratio with a zero denominator      -> flag divide-by-zero guard

Deliverables: a structured :class:`FeatureProfile` plus a rendered markdown
report and a JSON dump.

Example
-------
>>> from src.features.profiling import FeatureProfiler
>>> profile = FeatureProfiler().profile(X_train, y_train, X_test=X_test)
>>> profile.render_markdown("docs/day3_phaseA_feature_profile.md")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class Recommendation:
    """One candidate transformation with the evidence that motivates it."""

    feature: str
    transform: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureProfile:
    """Structured Phase-A result: per-block stats + the candidate-transform list."""

    n_train: int
    numerics: dict[str, Any] = field(default_factory=dict)
    categoricals: dict[str, Any] = field(default_factory=dict)
    correlations: list[dict[str, Any]] = field(default_factory=list)
    text: dict[str, Any] = field(default_factory=dict)
    dates: dict[str, Any] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    def render_markdown(self, path: str | Path) -> str:
        md = _render_markdown(self)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        return md


class FeatureProfiler:
    """Profile the training features and produce a candidate-transformation list."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.profiling")

        pa = self.config.get("day3", {}).get("profiling", {})
        self.skew_threshold: float = float(pa.get("skew_threshold", 1.0))
        self.onehot_max: int = int(pa.get("onehot_max_cardinality", 25))
        self.corr_threshold: float = float(pa.get("corr_threshold", 0.9))

        # Column roles (fall back to the known Day-3 layout).
        self.numeric_cols = pa.get("numeric", ["goal", "usd_goal_real"])
        self.categorical_cols = pa.get(
            "categorical", ["category", "main_category", "currency", "country"]
        )
        self.text_col = pa.get("text", "name_clean")
        self.date_cols = pa.get("dates", ["launched", "deadline"])

    # ------------------------------------------------------------------ public
    def profile(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> FeatureProfile:
        prof = FeatureProfile(n_train=int(len(X)))
        recs: list[Recommendation] = []

        self._profile_numerics(X, prof, recs)
        self._profile_categoricals(X, X_test, prof, recs)
        self._profile_correlations(X, prof, recs)
        self._profile_text(X, prof, recs)
        self._profile_dates(X, prof, recs)

        prof.recommendations = recs
        self.logger.info(
            "Feature profiling done — %d numerics, %d categoricals, %d recommendation(s).",
            len(prof.numerics),
            len(prof.categoricals),
            len(recs),
        )
        return prof

    # --------------------------------------------------------------- numerics
    def _profile_numerics(self, X, prof, recs) -> None:
        for c in self.numeric_cols:
            if c not in X.columns:
                continue
            s = pd.to_numeric(X[c], errors="coerce").astype(float)
            log_s = np.log1p(s.clip(lower=0))
            stats = {
                "min": float(s.min()),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "max": float(s.max()),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurt()),
                "log_skew": float(log_s.skew()),
                "log_kurtosis": float(log_s.kurt()),
                "n_zero": int((s == 0).sum()),
            }
            prof.numerics[c] = stats
            if abs(stats["skew"]) > self.skew_threshold:
                recs.append(
                    Recommendation(
                        feature=c,
                        transform="log1p",
                        reason=(
                            f"heavy right skew ({stats['skew']:.1f}) collapses to "
                            f"{stats['log_skew']:.2f} under log1p"
                        ),
                        evidence={"skew": stats["skew"], "log_skew": stats["log_skew"]},
                    )
                )

    # ----------------------------------------------------------- categoricals
    def _profile_categoricals(self, X, X_test, prof, recs) -> None:
        for c in self.categorical_cols:
            if c not in X.columns:
                continue
            vc = X[c].value_counts(dropna=False)
            card = int(X[c].nunique(dropna=True))
            entry = {
                "cardinality": card,
                "top": {str(k): int(v) for k, v in vc.head(5).items()},
            }
            # Informational only: unseen levels in test (handle_unknown sizing).
            if X_test is not None and c in X_test.columns:
                unseen = set(X_test[c].dropna().unique()) - set(X[c].dropna().unique())
                entry["unseen_in_test"] = len(unseen)
            prof.categoricals[c] = entry

            if card <= self.onehot_max:
                recs.append(
                    Recommendation(
                        feature=c,
                        transform="one_hot",
                        reason=f"low cardinality ({card} <= {self.onehot_max})",
                        evidence={"cardinality": card},
                    )
                )
            else:
                recs.append(
                    Recommendation(
                        feature=c,
                        transform="frequency_encode",
                        reason=f"high cardinality ({card} > {self.onehot_max}); "
                        "one-hot would explode width",
                        evidence={"cardinality": card},
                    )
                )

    # ---------------------------------------------------------- correlations
    def _profile_correlations(self, X, prof, recs) -> None:
        present = [c for c in self.numeric_cols if c in X.columns]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                sa = pd.to_numeric(X[a], errors="coerce").astype(float)
                sb = pd.to_numeric(X[b], errors="coerce").astype(float)
                pearson = float(sa.corr(sb))
                spearman = float(sa.corr(sb, method="spearman"))
                prof.correlations.append(
                    {"pair": [a, b], "pearson": pearson, "spearman": spearman}
                )
                if abs(pearson) > self.corr_threshold:
                    recs.append(
                        Recommendation(
                            feature=f"{a}|{b}",
                            transform="drop_one_of_pair",
                            reason=f"near-collinear (pearson {pearson:.3f}); keep one",
                            evidence={"pearson": pearson, "spearman": spearman},
                        )
                    )

    # ------------------------------------------------------------------- text
    def _profile_text(self, X, prof, recs) -> None:
        if self.text_col not in X.columns:
            return
        t = X[self.text_col].fillna("").astype(str)
        wc = t.str.split().str.len()
        ln = t.str.len()
        prof.text = {
            "column": self.text_col,
            "word_count": {
                "mean": float(wc.mean()),
                "median": float(wc.median()),
                "p95": float(wc.quantile(0.95)),
                "max": int(wc.max()),
            },
            "char_length": {
                "mean": float(ln.mean()),
                "median": float(ln.median()),
                "p95": float(ln.quantile(0.95)),
                "max": int(ln.max()),
            },
            "n_empty": int((wc == 0).sum()),
        }
        recs.append(
            Recommendation(
                feature=self.text_col,
                transform="tfidf_small_vocab",
                reason=(
                    f"short titles (~{wc.mean():.1f} words); cap TF-IDF features low "
                    "(~800) with (1,2)-grams, not 2500"
                ),
                evidence={"mean_words": float(wc.mean()), "p95_words": float(wc.quantile(0.95))},
            )
        )

    # ------------------------------------------------------------------ dates
    def _profile_dates(self, X, prof, recs) -> None:
        if not all(c in X.columns for c in self.date_cols):
            return
        launched = pd.to_datetime(X[self.date_cols[0]], errors="coerce")
        deadline = pd.to_datetime(X[self.date_cols[1]], errors="coerce")
        dur = (deadline - launched).dt.days
        n_zero_dur = int((dur <= 0).sum())
        prof.dates = {
            "campaign_duration": {
                "min": float(dur.min()),
                "median": float(dur.median()),
                "mean": float(dur.mean()),
                "p95": float(dur.quantile(0.95)),
                "max": float(dur.max()),
                "skew": float(dur.skew()),
                "n_nonpositive": n_zero_dur,
            },
            "n_unparseable": int(launched.isna().sum() + deadline.isna().sum()),
        }
        recs.append(
            Recommendation(
                feature="campaign_duration",
                transform="engineer_from_dates",
                reason="derive (deadline - launched).days + temporal parts (year/month/weekday)",
                evidence={"median_days": float(dur.median())},
            )
        )
        if n_zero_dur > 0:
            recs.append(
                Recommendation(
                    feature="goal_per_day",
                    transform="guard_divide_by_zero",
                    reason=(
                        f"{n_zero_dur} row(s) have duration <= 0; goal/duration would "
                        "divide by zero — clip denominator to >= 1"
                    ),
                    evidence={"n_nonpositive_duration": n_zero_dur},
                )
            )


# --------------------------------------------------------------------- render
def _render_markdown(p: FeatureProfile) -> str:
    lines: list[str] = []
    lines.append("# Day 3 — Phase A: Feature Profile (train-only)\n")
    lines.append(f"_Generated {p.created_utc} · train rows: {p.n_train:,}_\n")

    lines.append("## Numerics\n")
    lines.append("| feature | min | median | max | skew | kurtosis | log skew |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for c, s in p.numerics.items():
        lines.append(
            f"| `{c}` | {s['min']:.0f} | {s['median']:.0f} | {s['max']:.0f} | "
            f"{s['skew']:.2f} | {s['kurtosis']:.1f} | {s['log_skew']:.2f} |"
        )

    lines.append("\n## Categoricals\n")
    lines.append("| feature | cardinality | unseen in test | top levels |")
    lines.append("|---|--:|--:|---|")
    for c, e in p.categoricals.items():
        top = ", ".join(f"{k} ({v:,})" for k, v in list(e["top"].items())[:3])
        lines.append(
            f"| `{c}` | {e['cardinality']} | {e.get('unseen_in_test', '—')} | {top} |"
        )

    lines.append("\n## Numeric correlations\n")
    for c in p.correlations:
        lines.append(
            f"- `{c['pair'][0]}` ↔ `{c['pair'][1]}`: pearson **{c['pearson']:.3f}**, "
            f"spearman **{c['spearman']:.3f}**"
        )

    if p.text:
        t = p.text
        lines.append("\n## Title text\n")
        lines.append(
            f"- `{t['column']}` word count: mean **{t['word_count']['mean']:.2f}**, "
            f"median {t['word_count']['median']:.0f}, p95 {t['word_count']['p95']:.0f}, "
            f"max {t['word_count']['max']} · empty: {t['n_empty']}"
        )

    if p.dates:
        d = p.dates["campaign_duration"]
        lines.append("\n## Dates → campaign_duration\n")
        lines.append(
            f"- duration (days): min **{d['min']:.0f}**, median {d['median']:.0f}, "
            f"mean {d['mean']:.1f}, p95 {d['p95']:.0f}, max {d['max']:.0f}, skew {d['skew']:.2f}"
        )
        lines.append(f"- rows with duration ≤ 0 (divide-by-zero risk): **{d['n_nonpositive']}**")

    lines.append("\n## Candidate transformations\n")
    lines.append("| feature | transform | reason |")
    lines.append("|---|---|---|")
    for r in p.recommendations:
        lines.append(f"| `{r.feature}` | **{r.transform}** | {r.reason} |")

    return "\n".join(lines) + "\n"
