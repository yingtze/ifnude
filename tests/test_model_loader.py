"""Tests for ifnude.model.loader and ifnude.model.resolver."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


class TestResolveOutputs:
    def test_resolves_by_name(self):
        from ifnude.model.resolver import resolve_outputs
        from ifnude.model.constants import OUT_BOXES, OUT_SCORES, OUT_LABELS
        session = MagicMock()
        b = np.array([[[[1,2,3,4]]]], dtype=np.float32)
        s = np.array([[[0.9]]], dtype=np.float32)
        l = np.array([[0]], dtype="int32")
        outs = []
        for n in (OUT_BOXES, OUT_SCORES, OUT_LABELS):
            m = MagicMock(); m.name = n; outs.append(m)
        session.get_outputs.return_value = outs
        boxes, scores, labels = resolve_outputs(session, [b, s, l])
        assert np.array_equal(boxes, b)

    def test_raises_on_unresolvable_outputs(self):
        from ifnude.model.resolver import resolve_outputs
        from ifnude.exceptions import InvalidOutputError
        session = MagicMock()
        m = MagicMock(); m.name = "x"
        session.get_outputs.return_value = [m]
        with pytest.raises(InvalidOutputError):
            resolve_outputs(session, [np.array([1.0])])


class TestGetModel:
    @patch("ifnude.model.loader.ensure_assets")
    @patch("ifnude.model.loader.onnxruntime.InferenceSession")
    @patch("ifnude.model.loader.CLASSES_PATH")
    def test_returns_session_and_classes(self, mock_path, mock_session, mock_ensure):
        mock_path.read_text.return_value = "EXPOSED_BREAST_F\nEXPOSED_BREAST_M\n"
        from ifnude.model import loader
        loader.get_model.cache_clear()
        session, classes = loader.get_model()
        assert "EXPOSED_BREAST_F" in classes
        loader.get_model.cache_clear()