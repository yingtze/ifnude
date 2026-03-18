"""ifnude.api — public detection and censoring functions."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .types import ImageInput, Detection
from .model.loader import get_model
from .model.constants import IGNORED_LABELS
from .model.resolver import resolve_outputs
from .io.image import preprocess_image

logger = logging.getLogger(__name__)


def detect(
    img: ImageInput,
    mode: str = "default",
    min_prob: float | None = None,
) -> list[Detection]:
    """Detect NSFW body parts in a single image.

    Args:
        img:      File path (str/Path), PIL Image, or BGR numpy array.
        mode:     ``"default"`` (higher accuracy) or ``"fast"`` (~3× faster).
        min_prob: Confidence threshold. Defaults to 0.6 / 0.5 (fast).

    Returns:
        List of dicts with keys ``box``, ``score``, ``label``.

    Raises:
        ValueError: if *mode* is invalid.
    """
    if mode not in ("default", "fast"):
        raise ValueError(f"mode must be 'default' or 'fast', got {mode!r}")

    model, classes = get_model()

    if mode == "fast":
        image, scale = preprocess_image(img, min_side=480, max_side=800)
        min_prob = min_prob if min_prob is not None else 0.5
    else:
        image, scale = preprocess_image(img)
        min_prob = min_prob if min_prob is not None else 0.6

    raw = model.run(
        [o.name for o in model.get_outputs()],
        {model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
    )

    boxes, scores, labels = resolve_outputs(model, raw)
    boxes = boxes / scale

    results: list[Detection] = []
    for box, score, label_idx in zip(boxes[0], scores[0], labels[0]):
        if float(score) < min_prob:
            continue
        label_idx = int(label_idx)
        if label_idx < 0 or label_idx >= len(classes):
            logger.warning("label_idx %d out of range (%d classes); skipping",
                           label_idx, len(classes))
            continue
        label = classes[label_idx]
        if label in IGNORED_LABELS:
            continue
        results.append({
            "box":   box.astype(int).tolist(),
            "score": float(score),
            "label": label,
        })

    return results


def detect_batch(
    images: list[ImageInput],
    mode: str = "default",
    min_prob: float | None = None,
) -> list[list[Detection]]:
    """Run detection on a list of images reusing a single model instance.

    The ONNX session is initialised only once regardless of list length.
    Each image is still processed individually (not true ONNX batching).
    """
    return [detect(img, mode=mode, min_prob=min_prob) for img in images]


def censor(
    img: ImageInput,
    out_path: str | Path | None = None,
    visualize: bool = False,
    parts_to_blur: list[str] | None = None,
    detections: list[Detection] | None = None,
) -> np.ndarray:
    """Black-box censor detected NSFW regions in an image.

    Args:
        img:           File path or BGR numpy array.
        out_path:      Save path for censored image.
        visualize:     Display result with ``cv2.imshow``.
        parts_to_blur: Labels to censor; all detections by default.
        detections:    Pre-computed results from :func:`detect` — skips
                       inference if provided.

    Returns:
        Censored image as BGR numpy array.

    Raises:
        ValueError:        if neither *out_path* nor *visualize* is given.
        FileNotFoundError: if *img* path cannot be read.
    """
    if out_path is None and not visualize:
        raise ValueError("Provide at least one of out_path or visualize=True.")

    if isinstance(img, (str, Path)):
        image = cv2.imread(str(img))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img}")
    else:
        image = np.array(img)

    if detections is None:
        detections = detect(img)

    boxes = [
        d["box"] for d in detections
        if parts_to_blur is None or d["label"] in parts_to_blur
    ]

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)

    if out_path is not None:
        cv2.imwrite(str(out_path), image)

    if visualize:
        cv2.imshow("censored", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image