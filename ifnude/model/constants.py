"""ifnude.model.constants — URLs, filesystem paths, and label configuration."""
from __future__ import annotations
from pathlib import Path

# HuggingFace asset URLs
MODEL_URL   = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/detector.onnx"
CLASSES_URL = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/classes"

# Local cache directory
MODEL_FOLDER = Path.home() / ".ifnude"
MODEL_PATH   = MODEL_FOLDER / "detector.onnx"
CLASSES_PATH = MODEL_FOLDER / "classes"

# Labels the model emits that we never surface to callers
IGNORED_LABELS: frozenset[str] = frozenset({"EXPOSED_BELLY"})

# Actual ONNX output tensor names from the HuggingFace detector model
OUT_BOXES  = "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0"
OUT_SCORES = "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0"
OUT_LABELS = "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0"