"""Shared utilities for the AI Venture Intelligence Platform.

Provides cross-cutting helpers used by every module: structured logging
(:mod:`src.utils.logger`) and configuration loading (:mod:`src.utils.config`).
"""

from .config import load_config
from .logger import get_logger

__all__ = ["get_logger", "load_config"]
