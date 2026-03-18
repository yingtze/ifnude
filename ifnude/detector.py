"""Deprecated: use ifnude.api instead."""
import warnings
warnings.warn(
    "ifnude.detector is deprecated; use `from ifnude import detect` instead.",
    DeprecationWarning, stacklevel=2,
)
from .api import detect, detect_batch, censor  # noqa: F401
