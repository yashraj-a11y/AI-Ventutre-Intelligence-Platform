"""Phase G — Text Preprocessing (Day-3 Stage-2, Week 1).

Normalises the campaign title (``name_clean``) into a clean token string that
Phase H's TF-IDF vectorizer can turn into features. The pipeline collapses
surface variation so that *wallet*, *Wallets* and *WALLET* all become one token:

    lowercase -> strip non-alpha -> tokenize -> drop stopwords -> lemmatize

The point is **vocabulary reduction without meaning loss**: fewer, canonical
tokens mean a smaller, denser TF-IDF vocabulary and less noise. Titles are short
(~5.6 words) and weak signal, so aggressive normalisation helps the little
signal there is survive.

**Stateless / leakage-free.** The stopword list and lemmatizer are *fixed
linguistic resources*, not statistics learned from the data — so, like
:class:`~src.features.feature_builder.FeatureBuilder`, this transformer learns
nothing and cannot leak. (Phase H's idf/vocabulary *is* learned, and must fit on
train only.)

Dependency
----------
Requires NLTK ``stopwords`` and ``wordnet`` data. Install once with
``.venv/bin/python scripts/download_nltk_data.py``.

Example
-------
>>> tp = TextPreprocessor().fit(X_train)
>>> tp.process_text("Handcrafted Leather Wallets for Modern Adventurers")
'handcrafted leather wallet modern adventurer'
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

_TOKEN_RE = re.compile(r"[a-z]+")  # after lowercasing, words are runs of a-z


class TextPreprocessor:
    """Lowercase → strip → tokenize → de-stopword → lemmatize a text column.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.text_preprocessing")

        tc = self.config.get("day3", {}).get("text", {})
        self.input_col: str = tc.get("input_col", "name_clean")
        self.output_col: str = tc.get("output_col", "name_processed")
        self.min_token_length: int = int(tc.get("min_token_length", 2))
        self.remove_stopwords: bool = bool(tc.get("remove_stopwords", True))
        self.lemmatize: bool = bool(tc.get("lemmatize", True))
        self.keep_words: set[str] = set(tc.get("keep_words", []))

        self._stopwords: set[str] = set()
        self._lemmatizer = None
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: Any = None, y: Any = None) -> "TextPreprocessor":
        """Load linguistic resources (no data statistics learned → leakage-free)."""
        if self.remove_stopwords:
            from nltk.corpus import stopwords

            # NLTK English stopwords MINUS any domain words we choose to keep.
            self._stopwords = set(stopwords.words("english")) - self.keep_words
        if self.lemmatize:
            from nltk.stem import WordNetLemmatizer

            self._lemmatizer = WordNetLemmatizer()

        self._fitted = True
        self.logger.info(
            "TextPreprocessor ready — %d stopwords, lemmatize=%s, min_len=%d.",
            len(self._stopwords),
            self.lemmatize,
            self.min_token_length,
        )
        return self

    # -------------------------------------------------------------- transform
    def process_text(self, text: str) -> str:
        """Normalise a single string to a space-joined token string."""
        if not isinstance(text, str) or not text:
            return ""
        tokens = _TOKEN_RE.findall(text.lower())  # lowercase + strip non-alpha + tokenize

        out: list[str] = []
        for tok in tokens:
            if len(tok) < self.min_token_length:
                continue
            if self.remove_stopwords and tok in self._stopwords:
                continue
            if self.lemmatize:
                tok = self._lemma(tok)
            out.append(tok)
        return " ".join(out)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with ``output_col`` = processed ``input_col``."""
        self._check_fitted()
        out = X.copy()
        source = out[self.input_col].fillna("").astype(str)
        out[self.output_col] = source.map(self.process_text)

        n_empty = int((out[self.output_col].str.len() == 0).sum())
        self.logger.info(
            "Processed %d titles into '%s' (%d empty after cleaning).",
            len(out),
            self.output_col,
            n_empty,
        )
        return out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # --------------------------------------------------------------- internals
    @lru_cache(maxsize=100_000)
    def _lemma_cached(self, tok: str) -> str:
        # Cached because a ~50k-word vocabulary repeats across ~1.5M tokens.
        return self._lemmatizer.lemmatize(tok)

    def _lemma(self, tok: str) -> str:
        return self._lemma_cached(tok)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TextPreprocessor must be fit() before transform().")
