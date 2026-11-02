"""Configuration loading for the preprocessing pipeline.

Loads the YAML contract in ``configs/preprocessing.yaml`` — target states,
dropped/leakage columns, missing-value strategy, integrity thresholds, datatype
targets — so that preprocessing behaviour is configuration-driven rather than
hard-coded. Stage-2 keys (scaling/encoding/TF-IDF) live under ``day3_reserved``
and are not consumed in Day 2.

Example
-------
>>> from src.utils.config import load_config
>>> cfg = load_config()
>>> cfg["target"]["decided_states"]
['successful', 'failed']
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml

# Repo root = two levels up from this file (src/utils/config.py -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "preprocessing.yaml"

# Day-18 deployment/runtime settings live in ENV VARS (12-factor), separate from the ML YAML above.
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_MODES = {"development", "staging", "production"}


def load_config(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load and parse the preprocessing YAML configuration.

    Parameters
    ----------
    path:
        Path to the config file. Defaults to ``configs/preprocessing.yaml``
        relative to the repository root.

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the file is empty or does not parse to a mapping.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration at {config_path} did not parse to a mapping."
        )

    return config


def deployment_settings(environ: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Day-18 deployment/runtime settings from environment variables (12-factor).

    Separate from :func:`load_config` (which owns the ML YAML) — this reads *deployment* config:
    ports, paths, logging, mode, workers, CORS. **There are NO secrets/API keys** (the platform is
    deterministic with no external services), so none are read or required.

    Every setting has a safe default, so nothing is strictly *required*; validation instead rejects
    malformed *values* (a bad ``LOG_LEVEL`` or a non-integer ``PORT``) with a clear ``ValueError`` —
    fail fast at startup, not deep in a request.
    """
    env = os.environ if environ is None else environ

    def _int(name: str, default: int) -> int:
        raw = env.get(name)
        if raw in (None, ""):
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer, got {raw!r}")

    log_level = env.get("LOG_LEVEL", "INFO").upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {log_level!r}")

    mode = env.get("DEPLOY_MODE", "production").lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"DEPLOY_MODE must be one of {sorted(_VALID_MODES)}, got {mode!r}")

    # Comma-separated CORS origins; empty => no cross-origin access (safe default).
    origins = [o.strip() for o in env.get("CORS_ORIGINS", "").split(",") if o.strip()]

    return {
        "port": _int("PORT", 8000),
        "workers": _int("WORKERS", 1),
        "log_level": log_level,
        "log_file": env.get("LOG_FILE") or None,
        "deploy_mode": mode,
        "artifacts_dir": env.get("ARTIFACTS_DIR", "artifacts"),
        "data_dir": env.get("DATA_DIR", "data"),
        "cors_origins": origins,
    }
