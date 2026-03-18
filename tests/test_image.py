"""Tests for ifnude.io.image."""
from __future__ import annotations
import numpy as np
import pytest
from ifnude.io.image import compute_resize_scale, resize_image, preprocess_image


class TestComputeResizeScale:
    def test_capped_by_max_side(self):
        scale = compute_resize_scale((100, 200, 3), min_side=800, max_side=1333)
        assert scale == pytest.approx(1333 / 200)

    def test_capped_landscape(self):
        scale = compute_resize_scale((400, 2000, 3), min_side=800, max_side=1333)
        assert scale == pytest.approx(1333 / 2000)

    def test_no_scaling_needed(self):
        assert compute_resize_scale((800, 1000, 3), min_side=800, max_side=1333) == pytest.approx(1.0)


class TestResizeImage:
    def test_output_dimensions(self):
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        resized, scale = resize_image(img, min_side=800, max_side=1333)
        assert resized.shape[0] == pytest.approx(400 * scale, abs=1)


class TestPreprocessImage:
    def test_numpy_input_returns_float32(self):
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        out, scale = preprocess_image(img)
        assert out.dtype == np.float32
        assert isinstance(scale, float)

    def test_invalid_type_raises(self):
        with pytest.raises(NotImplementedError):
            preprocess_image(12345)  # type: ignore
