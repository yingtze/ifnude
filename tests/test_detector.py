"""Unit tests for ifnude – no model download required."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from ifnude.detector_utils import compute_resize_scale, resize_image, preprocess_image


# ---------------------------------------------------------------------------
# detector_utils
# ---------------------------------------------------------------------------

class TestComputeResizeScale:
    def test_capped_by_max_side(self):
        # min_side=800, scale=8 → largest 200*8=1600 > 1333 → capped
        scale = compute_resize_scale((100, 200, 3), min_side=800, max_side=1333)
        assert scale == pytest.approx(1333 / 200)

    def test_capped_landscape(self):
        scale = compute_resize_scale((400, 2000, 3), min_side=800, max_side=1333)
        assert scale == pytest.approx(1333 / 2000)

    def test_no_scale_needed(self):
        assert compute_resize_scale((800, 1000, 3), min_side=800, max_side=1333) == pytest.approx(1.0)


class TestResizeImage:
    def test_output_scaled_correctly(self):
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        resized, scale = resize_image(img, min_side=800, max_side=1333)
        assert resized.shape[0] == pytest.approx(400 * scale, abs=1)


class TestPreprocessImage:
    def test_accepts_numpy_array(self):
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        out, scale = preprocess_image(img)
        assert isinstance(out, np.ndarray) and isinstance(scale, float)

    def test_no_pil_roundtrip_for_numpy(self):
        """numpy input should not produce dtype object (no PIL round-trip)."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out, _ = preprocess_image(img)
        assert out.dtype == np.float32

    def test_raises_on_invalid_type(self):
        with pytest.raises(NotImplementedError):
            preprocess_image(12345)  # type: ignore


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------

def _mock_session(outputs):
    session = MagicMock()
    session.get_outputs.return_value = [
        MagicMock(name="boxes"),
        MagicMock(name="scores"),
        MagicMock(name="labels"),
    ]
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.run.return_value = outputs
    return session


class TestDetect:
    @patch("ifnude.detector._get_model")
    def test_returns_list_of_dicts(self, mock_get):
        session = _mock_session([
            np.array([[[[10, 20, 50, 60]]]], dtype=np.float32),  # boxes
            np.array([[[0.9]]], dtype=np.float32),               # scores
            np.array([[0]], dtype="int32"),                      # labels
        ])
        mock_get.return_value = (session, ["EXPOSED_BREAST_F"])
        from ifnude.detector import detect
        result = detect(np.zeros((300, 400, 3), dtype=np.uint8))
        assert result
        assert all("box" in r and "score" in r and "label" in r for r in result)

    @patch("ifnude.detector._get_model")
    def test_min_prob_filters_low_score(self, mock_get):
        session = _mock_session([
            np.array([[[[10, 10, 50, 50]]]], dtype=np.float32),
            np.array([[[0.3]]], dtype=np.float32),
            np.array([[0]], dtype="int32"),
        ])
        mock_get.return_value = (session, ["EXPOSED_BREAST_F"])
        from ifnude.detector import detect
        assert detect(np.zeros((300, 400, 3), dtype=np.uint8), min_prob=0.95) == []

    def test_invalid_mode_raises(self):
        from ifnude.detector import detect
        with pytest.raises(ValueError, match="mode"):
            detect(np.zeros((10, 10, 3), dtype=np.uint8), mode="turbo")

    @patch("ifnude.detector._get_model")
    def test_out_of_range_label_skipped(self, mock_get):
        """label_idx beyond classes list should be skipped, not crash."""
        session = _mock_session([
            np.array([[[[10, 10, 50, 50]]]], dtype=np.float32),
            np.array([[[0.9]]], dtype=np.float32),
            np.array([[99]], dtype="int32"),  # out of range
        ])
        mock_get.return_value = (session, ["EXPOSED_BREAST_F"])  # only 1 class
        from ifnude.detector import detect
        result = detect(np.zeros((300, 400, 3), dtype=np.uint8))
        assert result == []


# ---------------------------------------------------------------------------
# censor()
# ---------------------------------------------------------------------------

class TestCensor:
    def test_raises_when_no_output(self):
        from ifnude.detector import censor
        with pytest.raises(ValueError):
            censor(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_empty_out_path_string_raises(self):
        """Empty string out_path should also raise, not silently pass."""
        from ifnude.detector import censor
        with pytest.raises(ValueError):
            censor(np.zeros((10, 10, 3), dtype=np.uint8), out_path=None)

    @patch("ifnude.detector.detect")
    def test_precomputed_detections_skips_detect(self, mock_detect):
        """When detections= is supplied, detect() must not be called."""
        from ifnude.detector import censor
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        censor(img, visualize=False, out_path="/tmp/test_out.jpg", detections=[])
        mock_detect.assert_not_called()
