"""ifnude.io.image — image loading and preprocessing for model inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..types import ImageInput


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read *path* and return a BGR numpy array."""
    image = np.ascontiguousarray(Image.open(path).convert("RGB"))
    return image[:, :, ::-1]


def compute_resize_scale(
    image_shape: tuple[int, int, int],
    min_side: int = 800,
    max_side: int = 1333,
) -> float:
    rows, cols, _ = image_shape
    scale = min_side / min(rows, cols)
    if max(rows, cols) * scale > max_side:
        scale = max_side / max(rows, cols)
    return scale


def resize_image(
    img: np.ndarray,
    min_side: int = 800,
    max_side: int = 1333,
) -> tuple[np.ndarray, float]:
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
    return cv2.resize(img, None, fx=scale, fy=scale), scale


def _normalize(x: np.ndarray, mode: str = "caffe") -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "tf":
        x /= 127.5
        x -= 1.0
    elif mode == "caffe":
        x -= [103.939, 116.779, 123.68]
    return x


def preprocess_image(
    img: ImageInput,
    min_side: int = 800,
    max_side: int = 1333,
) -> tuple[np.ndarray, float]:
    """Normalise and resize *img* ready for ONNX inference.

    Args:
        img:      File path (str/Path), PIL Image, or BGR numpy array.
        min_side: Target minimum side length.
        max_side: Target maximum side length.

    Returns:
        (preprocessed_array, scale_factor)

    Raises:
        NotImplementedError: if *img* type is unsupported.
    """
    if isinstance(img, (str, Path)):
        image = read_image_bgr(img)
    elif isinstance(img, Image.Image):
        image = np.ascontiguousarray(img.convert("RGB"))[:, :, ::-1]
    elif isinstance(img, np.ndarray):
        image = np.ascontiguousarray(img)
    else:
        raise NotImplementedError(
            f"img must be a file path, PIL Image, or numpy array; got {type(img)}"
        )

    image, scale = resize_image(_normalize(image), min_side=min_side, max_side=max_side)
    return image, scale
