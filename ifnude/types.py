"""ifnude.types — shared type aliases used across the package."""
from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, "Image.Image"]
Detection  = dict  # {"box": list[int], "score": float, "label": str}
