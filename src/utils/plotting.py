"""Lazy, headless matplotlib for modules that only plot offline.

Training/evaluation/explainability modules draw PNGs, but the serving path imports the same
packages without ever plotting. Importing pyplot at module load cost memory on every API start
(and made matplotlib a hard serving dependency). ``plt`` here is a stand-in that imports
``matplotlib`` — with the same ``Agg`` backend as before — on first attribute access.

Example
-------
>>> from src.utils.plotting import plt
>>> fig, ax = plt.subplots()      # matplotlib is imported here, not at module import
"""

from __future__ import annotations

from typing import Any


class _LazyPyplot:
    """Proxy for ``matplotlib.pyplot``; the real module loads on first use."""

    _mod: Any = None

    def __getattr__(self, name: str) -> Any:
        if _LazyPyplot._mod is None:
            import matplotlib
            matplotlib.use("Agg")                # headless: render to file, never to a window
            import matplotlib.pyplot as pyplot
            _LazyPyplot._mod = pyplot
        return getattr(_LazyPyplot._mod, name)


plt = _LazyPyplot()
