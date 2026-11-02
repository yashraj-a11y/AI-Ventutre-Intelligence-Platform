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
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

_DEFAULT_FORMAT = "%(asctime)s | %(name)-32s | %(levelname)-8s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Track configured loggers so we never attach duplicate handlers.
_CONFIGURED: set[str] = set()

# Day-18: a process-wide shared file handler + level, set by configure_logging(). While unset, the
# module behaves exactly as before (console only, INFO) — so nothing changes unless it's opted into.
_FILE_HANDLER: Optional[logging.Handler] = None
_ACTIVE_LEVEL: Optional[Union[int, str]] = None


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
    effective_level = _ACTIVE_LEVEL if _ACTIVE_LEVEL is not None else level

    if name in _CONFIGURED:
        # Already configured; just make sure the level is current and return.
        logger.setLevel(effective_level)
        return logger

    logger.setLevel(effective_level)
    logger.propagate = False  # avoid double logging via the root logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:                       # explicit per-call file sink
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    elif _FILE_HANDLER is not None:                # the process-wide sink from configure_logging()
        logger.addHandler(_FILE_HANDLER)

    _CONFIGURED.add(name)
    return logger


def configure_logging(log_file: Optional[Union[str, Path]] = None,
                      level: Optional[Union[int, str]] = None,
                      environ: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Day-18 — attach a process-wide file sink + level to ALL ``avip`` loggers (existing + future).

    Reads ``LOG_FILE`` / ``LOG_LEVEL`` from the environment when not passed. Idempotent: the shared
    :class:`FileHandler` is created once and shared by every logger (so one file, no duplicates). It
    retroactively applies to loggers already created at import time, and :func:`get_logger` attaches it
    to any created later — so a single call at startup centralizes the whole package's logs.
    """
    global _FILE_HANDLER, _ACTIVE_LEVEL
    env = os.environ if environ is None else environ
    log_file = log_file or env.get("LOG_FILE") or None
    _ACTIVE_LEVEL = level or env.get("LOG_LEVEL", "INFO")

    if log_file and _FILE_HANDLER is None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        _FILE_HANDLER = logging.FileHandler(path, encoding="utf-8")
        _FILE_HANDLER.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))

    n = 0
    for lname in list(logging.Logger.manager.loggerDict):      # already-created avip loggers
        if lname.startswith("avip"):
            lg = logging.getLogger(lname)
            lg.setLevel(_ACTIVE_LEVEL)
            if _FILE_HANDLER is not None and _FILE_HANDLER not in lg.handlers:
                lg.addHandler(_FILE_HANDLER)
            n += 1
    return {"log_file": str(log_file) if log_file else None, "level": _ACTIVE_LEVEL,
            "loggers_reconfigured": n}
