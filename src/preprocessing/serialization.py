"""
serialization.py

Production-ready serialization utilities for the
AI Venture Intelligence Platform.

Author: Yash
"""

from pathlib import Path
from typing import Any

import joblib


def _validate_filepath(filepath: str | Path) -> Path:
    """
    Validate and convert the provided filepath into a Path object.

    Parameters
    ----------
    filepath : str | Path
        Destination or source path.

    Returns
    -------
    pathlib.Path
        Validated Path object.
    """

    if not isinstance(filepath, (str, Path)):
        raise TypeError(
            "filepath must be of type str or pathlib.Path."
        )

    return Path(filepath)


def save_object(
    obj: Any,
    filepath: str | Path,
    compress: int = 3,
) -> None:
    """
    Save any Python object using joblib.

    Parameters
    ----------
    obj : Any
        Python object to serialize.

    filepath : str | Path
        Destination file.

    compress : int, default=3
        Compression level (0-9).

    Returns
    -------
    None
    """

    filepath = _validate_filepath(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        joblib.dump(
            obj,
            filepath,
            compress=compress,
        )

    except Exception as error:

        raise RuntimeError(
            f"Failed to save object to '{filepath}'."
        ) from error


def load_object(
    filepath: str | Path,
) -> Any:
    """
    Load a serialized Python object.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    Any
        Loaded object.
    """

    filepath = _validate_filepath(filepath)

    if not filepath.exists():

        raise FileNotFoundError(
            f"File does not exist: {filepath}"
        )

    try:
        return joblib.load(filepath)

    except Exception as error:

        raise RuntimeError(
            f"Failed to load object from '{filepath}'."
        ) from error


def save_pipeline(
    pipeline: Any,
    filepath: str | Path,
) -> None:
    """
    Save a fitted preprocessing pipeline.

    Parameters
    ----------
    pipeline : Any
        Fitted PreprocessingPipeline.

    filepath : str | Path

    Returns
    -------
    None
    """

    save_object(
        obj=pipeline,
        filepath=filepath,
    )


def load_pipeline(
    filepath: str | Path,
) -> Any:
    """
    Load a fitted preprocessing pipeline.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    Any
        Loaded preprocessing pipeline.
    """

    return load_object(filepath)