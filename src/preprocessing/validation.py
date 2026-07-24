
"""Phase A — Data Validation Layer.

Before any cleaning happens, :class:`DatasetValidator` verifies that the incoming
dataframe is *structurally* valid against the frozen 15-column Kickstarter schema.
This is schema/structure validation only — business-rule checks (impossible goals,
bad dates, the ``N,0"`` cohort) live in Phase F (:mod:`src.preprocessing.integrity`).

Checks performed
----------------
- Empty-dataset detection (no rows or no columns)              -> CRITICAL
- Duplicate column names                                       -> CRITICAL
- Missing required columns (against the 15-column schema)      -> CRITICAL
- Unexpected/extra columns                                     -> WARNING
- Invalid ``state`` target values                              -> ERROR
- Identifier (``ID``) uniqueness                               -> ERROR

The validator never mutates the dataframe. It returns a :class:`ValidationReport`
and (optionally) raises :class:`DataValidationError` on a failing dataset.

Example
-------
>>> from src.preprocessing.validation import DatasetValidator
>>> validator = DatasetValidator()
>>> report = validator.validate(df)
>>> report.passed
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

# Severity ordering: CRITICAL/ERROR fail the dataset; WARNING does not.
_CRITICAL = "CRITICAL"
_ERROR = "ERROR"
_WARNING = "WARNING"
_PASS = "PASS"

_FAILING = {_CRITICAL, _ERROR}


class DataValidationError(Exception):
    """Raised when a dataset fails validation and strict handling is requested."""


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    status: str  # PASS | WARNING | ERROR | CRITICAL
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Structured, JSON-serializable result of dataset validation."""

    passed: bool
    n_rows: int
    n_columns: int
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status in _FAILING]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == _WARNING]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetValidator:
    """Validate a raw dataframe against the frozen schema contract.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.preprocessing.validation")

        schema = self.config.get("schema", {})
        self.required_columns: list[str] = list(schema.get("required_columns", []))

        target = self.config.get("target", {})
        self.target_column: str = target.get("source_column", "state")
        self.valid_states: list[str] = list(
            self.config.get("integrity", {}).get("valid_states", [])
        )

        # Identifier column (first entry of columns.identifier, default "ID").
        identifiers = self.config.get("columns", {}).get("identifier", ["ID"])
        self.id_column: str = identifiers[0] if identifiers else "ID"

    # ------------------------------------------------------------------ public
    def validate(
        self, df: pd.DataFrame, raise_on_fail: bool = False
    ) -> ValidationReport:
        """Run all structural checks and return a :class:`ValidationReport`.

        Parameters
        ----------
        df:
            The raw dataframe to validate (not modified).
        raise_on_fail:
            If ``True``, raise :class:`DataValidationError` when the dataset
            fails (any CRITICAL or ERROR check).
        """
        checks: list[CheckResult] = []

        # 1. Empty dataset — short-circuit; nothing else is meaningful.
        empty = self._check_empty(df)
        checks.append(empty)
        if empty.status == _CRITICAL:
            report = ValidationReport(
                passed=False, n_rows=len(df), n_columns=df.shape[1], checks=checks
            )
            self._log_report(report)
            if raise_on_fail:
                raise DataValidationError(empty.message)
            return report

        # 2–6. Structural + content checks.
        checks.append(self._check_duplicate_columns(df))
        checks.append(self._check_required_columns(df))
        checks.append(self._check_unexpected_columns(df))
        checks.append(self._check_target_values(df))
        checks.append(self._check_id_uniqueness(df))

        passed = not any(c.status in _FAILING for c in checks)
        report = ValidationReport(
            passed=passed, n_rows=len(df), n_columns=df.shape[1], checks=checks
        )
        self._log_report(report)

        if raise_on_fail and not passed:
            failing = "; ".join(f"[{c.status}] {c.name}: {c.message}" for c in report.errors)
            raise DataValidationError(f"Dataset validation failed — {failing}")

        return report

    def assert_valid(self, df: pd.DataFrame) -> ValidationReport:
        """Validate and raise :class:`DataValidationError` on failure."""
        return self.validate(df, raise_on_fail=True)

    # ------------------------------------------------------------------ checks
    def _check_empty(self, df: pd.DataFrame) -> CheckResult:
        if df is None or df.shape[0] == 0 or df.shape[1] == 0:
            return CheckResult(
                "empty_dataset",
                _CRITICAL,
                "Dataset is empty (no rows or no columns).",
                {"n_rows": 0 if df is None else int(df.shape[0])},
            )
        return CheckResult(
            "empty_dataset", _PASS, f"Dataset has {len(df):,} rows.", {"n_rows": len(df)}
        )

    def _check_duplicate_columns(self, df: pd.DataFrame) -> CheckResult:
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        if dupes:
            return CheckResult(
                "duplicate_columns",
                _CRITICAL,
                f"Duplicate column names found: {dupes}",
                {"duplicates": dupes},
            )
        return CheckResult("duplicate_columns", _PASS, "No duplicate column names.")

    def _check_required_columns(self, df: pd.DataFrame) -> CheckResult:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            return CheckResult(
                "required_columns",
                _CRITICAL,
                f"Missing {len(missing)} required column(s): {missing}",
                {"missing": missing},
            )
        return CheckResult(
            "required_columns",
            _PASS,
            f"All {len(self.required_columns)} required columns present.",
        )

    def _check_unexpected_columns(self, df: pd.DataFrame) -> CheckResult:
        extra = [c for c in df.columns if c not in self.required_columns]
        if extra:
            return CheckResult(
                "unexpected_columns",
                _WARNING,
                f"Unexpected column(s) present (not in schema): {extra}",
                {"unexpected": extra},
            )
        return CheckResult("unexpected_columns", _PASS, "No unexpected columns.")

    def _check_target_values(self, df: pd.DataFrame) -> CheckResult:
        if self.target_column not in df.columns:
            return CheckResult(
                "target_values",
                _ERROR,
                f"Target column '{self.target_column}' is absent.",
                {},
            )
        if not self.valid_states:
            return CheckResult(
                "target_values", _PASS, "No valid_states configured; skipped."
            )
        observed = set(df[self.target_column].dropna().unique())
        invalid = sorted(observed - set(self.valid_states))
        if invalid:
            return CheckResult(
                "target_values",
                _ERROR,
                f"Target '{self.target_column}' contains invalid states: {invalid}",
                {"invalid_states": invalid},
            )
        return CheckResult(
            "target_values",
            _PASS,
            f"All '{self.target_column}' values are within the valid set.",
        )

    def _check_id_uniqueness(self, df: pd.DataFrame) -> CheckResult:
        if self.id_column not in df.columns:
            return CheckResult(
                "id_uniqueness",
                _WARNING,
                f"Identifier column '{self.id_column}' is absent; uniqueness not checked.",
                {},
            )
        n_dupes = int(df[self.id_column].duplicated().sum())
        if n_dupes > 0:
            return CheckResult(
                "id_uniqueness",
                _ERROR,
                f"Identifier '{self.id_column}' has {n_dupes:,} duplicate value(s).",
                {"duplicate_ids": n_dupes},
            )
        return CheckResult(
            "id_uniqueness", _PASS, f"Identifier '{self.id_column}' is unique."
        )

    # ----------------------------------------------------------------- logging
    def _log_report(self, report: ValidationReport) -> None:
        for check in report.checks:
            line = f"[{check.name}] {check.status}: {check.message}"
            if check.status == _PASS:
                self.logger.info(line)
            elif check.status == _WARNING:
                self.logger.warning(line)
            else:
                self.logger.error(line)
        verdict = "PASSED" if report.passed else "FAILED"
        self.logger.info(
            "Validation %s — %d rows, %d columns, %d error(s), %d warning(s).",
            verdict,
            report.n_rows,
            report.n_columns,
            len(report.errors),
            len(report.warnings),
        )
