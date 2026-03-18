"""ifnude.model.loader — lazy ONNX session singleton."""
from __future__ import annotations

import functools
import logging

import onnxruntime

from .constants import MODEL_PATH, CLASSES_PATH
from ..exceptions import ModelLoadError
from ..io.downloader import ensure_assets

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_model() -> tuple[onnxruntime.InferenceSession, list[str]]:
    """Return a cached (session, classes) tuple.

    Downloads assets on first call if not already cached.
    Raises:
        ModelLoadError: if the ONNX session cannot be initialised.
    """
    ensure_assets()

    available = onnxruntime.get_available_providers()
    providers  = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                  if p in available] or ["CPUExecutionProvider"]
    try:
        session = onnxruntime.InferenceSession(str(MODEL_PATH), providers=providers)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load ONNX model from {MODEL_PATH}") from exc

    classes = [l.strip() for l in CLASSES_PATH.read_text().splitlines() if l.strip()]
    logger.debug("Model loaded with providers: %s, classes: %d", providers, len(classes))
    return session, classes
