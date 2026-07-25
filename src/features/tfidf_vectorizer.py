"""Phase H — TF-IDF Vectorization (Day-3 Stage-2, Week 2).

Turns the normalised titles (``name_processed`` from Phase G) into a numeric,
sparse feature matrix. Each column is a term (a 1- or 2-gram); each cell is the
term's **TF-IDF** weight: high when a term is frequent in *this* title but rare
across the corpus, low when it is common everywhere.

**First stateful text step.** Unlike Phase G (fixed stopwords/lexicon), TF-IDF
*learns* two things from the data — the **vocabulary** (which ~800 terms become
columns) and the **idf** weights (how rare each term is corpus-wide). Both are
statistics of the training set, so they must be fit on **train only**; a term
that appears only in the test set is simply ignored (it is not in the learned
vocabulary). Fitting on the full set would leak test-corpus statistics into the
weights.

**Sizing (frozen).** ``max_features≈800`` with ``ngram_range=(1,2)``. Titles are
~5.6 words, so the informative vocabulary is small; 2500 features (the stale
config value) would be mostly near-empty columns. 800 keeps the head of the term
distribution; bigrams catch short phrases ("board game", "art print").

Output is a :class:`scipy.sparse.csr_matrix` (N × ~800) — each title lights up
only a handful of the 800 columns, so the matrix is >99% zeros and must stay
sparse.

Example
-------
>>> vec = TfidfVectorizerHandler().fit(X_train)
>>> Xt = vec.transform(X_train)          # scipy.sparse.csr_matrix (n, ~800)
>>> Xt.shape[1]
800
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import load_config
from src.utils.logger import get_logger


class TfidfVectorizerHandler:
    """Fit a TF-IDF vectorizer on train titles; transform any partition to sparse.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.tfidf")

        tc = self.config.get("day3", {}).get("tfidf", {})
        self.input_col: str = tc.get("input_col", "name_processed")
        self.max_features: int = int(tc.get("max_features", 800))
        ngram = tc.get("ngram_range", [1, 2])
        self.ngram_range: tuple[int, int] = (int(ngram[0]), int(ngram[1]))
        self.min_df: int = int(tc.get("min_df", 2))
        self.max_df: float = float(tc.get("max_df", 0.95))
        self.sublinear_tf: bool = bool(tc.get("sublinear_tf", False))

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "TfidfVectorizerHandler":
        """Learn the vocabulary + idf weights from the **training** titles only."""
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )
        self._vectorizer.fit(self._text(X))
        self._fitted = True
        self.logger.info(
            "TF-IDF fitted — %d features, ngram=%s, min_df=%d, max_df=%.2f.",
            len(self._vectorizer.vocabulary_),
            self.ngram_range,
            self.min_df,
            self.max_df,
        )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        """Transform titles into a sparse TF-IDF matrix using the fitted vocabulary."""
        self._check_fitted()
        return self._vectorizer.transform(self._text(X)).tocsr()

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> sp.csr_matrix:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------ names
    def get_feature_names_out(self) -> list[str]:
        self._check_fitted()
        return self._vectorizer.get_feature_names_out().tolist()

    # -------------------------------------------------------------- persistence
    def save(
        self, path: str | Path = "artifacts/vectorizers/tfidf_vectorizer.joblib"
    ) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved fitted TF-IDF vectorizer to %s", path)
        return path

    @staticmethod
    def load(
        path: str | Path = "artifacts/vectorizers/tfidf_vectorizer.joblib",
    ) -> "TfidfVectorizerHandler":
        return joblib.load(Path(path))

    # --------------------------------------------------------------- internals
    def _text(self, X: pd.DataFrame) -> pd.Series:
        return X[self.input_col].fillna("").astype(str)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TfidfVectorizerHandler must be fit() before transform().")
