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

from pathlib import Path
from typing import Any, Optional, Union

import yaml

# Repo root = two levels up from this file (src/utils/config.py -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "preprocessing.yaml"


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
