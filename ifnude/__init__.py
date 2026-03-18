"""ifnude — nudity detection that just works."""

from .api import detect, detect_batch, censor

__version__ = "1.0.0"
__all__ = ["detect", "detect_batch", "censor"]
