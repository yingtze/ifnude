"""ifnude.exceptions — custom exception hierarchy."""
from __future__ import annotations


class IfnudeError(Exception):
    """Base exception for all ifnude errors."""


class AssetDownloadError(IfnudeError):
    """Raised when model or class-list download fails."""


class ModelLoadError(IfnudeError):
    """Raised when the ONNX session cannot be initialised."""


class InvalidOutputError(IfnudeError):
    """Raised when ONNX model outputs cannot be resolved."""
