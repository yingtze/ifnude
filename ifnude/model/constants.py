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

# Expected ONNX output tensor names (resolve by name, not by dtype)
OUT_BOXES  = "boxes"
OUT_SCORES = "scores"
OUT_LABELS = "labels"
