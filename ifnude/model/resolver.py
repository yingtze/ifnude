"""ifnude.model.resolver — map raw ONNX outputs to (boxes, scores, labels)."""
from __future__ import annotations

import logging

import numpy as np
import onnxruntime

from .constants import OUT_BOXES, OUT_SCORES, OUT_LABELS
from ..exceptions import InvalidOutputError

logger = logging.getLogger(__name__)


def resolve_outputs(
    session: onnxruntime.InferenceSession,
    raw: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (boxes, scores, labels) from raw ONNX output list.

    Resolves by tensor name first; falls back to dtype heuristics with a
    warning so maintainers notice when the model changes.

    Raises:
        InvalidOutputError: if outputs cannot be resolved either way.
    """
    name_map = {o.name: raw[i] for i, o in enumerate(session.get_outputs())}

    if all(k in name_map for k in (OUT_BOXES, OUT_SCORES, OUT_LABELS)):
        return name_map[OUT_BOXES], name_map[OUT_SCORES], name_map[OUT_LABELS]

    logger.warning(
        "ONNX output names %s do not match expected (%s, %s, %s). "
        "Falling back to dtype-based resolution — verify model outputs.",
        list(name_map.keys()), OUT_BOXES, OUT_SCORES, OUT_LABELS,
    )
    try:
        labels = next(op for op in raw if op.dtype == "int32")
        scores = next(op for op in raw if isinstance(op[0][0], np.float32))
        boxes  = next(op for op in raw if isinstance(op[0][0], np.ndarray))
    except StopIteration as exc:
        raise InvalidOutputError(
            f"Cannot resolve ONNX outputs. Got names: {list(name_map.keys())}"
        ) from exc

    return boxes, scores, labels
