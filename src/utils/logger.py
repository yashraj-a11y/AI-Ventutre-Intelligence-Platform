"""Centralized logging for the preprocessing pipeline.

Every preprocessing stage logs through :func:`get_logger` so that execution time,
rows removed, columns dropped, warnings, and errors are recorded in one uniform
format across the whole package. Handlers are attached once per logger name to
avoid duplicate output when the function is called repeatedly (e.g. per module).

Example
-------
>>> from src.utils.logger import get_logger
>>> log = get_logger("avip.preprocessing.validation")
>>> log.info("schema validation passed")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_DEFAULT_FORMAT = "%(asctime)s | %(name)-32s | %(levelname)-8s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Track configured loggers so we never attach duplicate handlers.
_CONFIGURED: set[str] = set()


def get_logger(
    name: str = "avip.preprocessing",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name. Use a dotted namespace per module, e.g.
        ``"avip.preprocessing.integrity"``.
    level:
        Logging level (``logging.INFO`` or a string like ``"DEBUG"``).
    log_file:
        Optional path to also write logs to a file. Parent directories are
        created automatically.
    fmt, datefmt:
        Log line and timestamp formats.

    Returns
    -------
    logging.Logger
        A logger with a stdout handler (and optional file handler) attached
        exactly once.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED:
        # Already configured; just make sure the level is current and return.
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False  # avoid double logging via the root logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger
