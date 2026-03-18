"""Deprecated: use ifnude.io.image instead."""
import warnings
warnings.warn(
    "ifnude.detector_utils is deprecated; use ifnude.io.image instead.",
    DeprecationWarning, stacklevel=2,
)
from .io.image import (  # noqa: F401
    read_image_bgr,
    compute_resize_scale,
    resize_image,
    preprocess_image,
)
