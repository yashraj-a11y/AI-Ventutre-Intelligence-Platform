"""Preprocessing package (Day 2 — Stage 1: dataset cleaning).

This package holds the row-changing "dataset cleaning" stage of the pipeline:
validation, cleaning/column-removal, target building, missing-value handling,
duplicate handling, datatype correction, and business-integrity validation —
orchestrated by :class:`~src.preprocessing.pipeline.PreprocessingPipeline`.

Stage 2 (row-preserving feature transforms: encoding, scaling, TF-IDF) is built
in Day 3 and lives outside this package.

Public handler classes are exported here as each Day 2 phase (A–H) lands. The
imports are guarded so the package remains importable while modules are still
being implemented.
"""

__all__: list[str] = []

# Exports are wired incrementally as each phase's module is implemented. Guarded
# so `import src.preprocessing` never fails on a not-yet-created module.
try:  # Phase A
    from .validation import DatasetValidator  # noqa: F401

    __all__.append("DatasetValidator")
except ImportError:
    pass
