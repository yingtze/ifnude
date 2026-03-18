"""Tests for ifnude.api — detect(), detect_batch(), censor()."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


def _mock_session(boxes, scores, labels):
    session = MagicMock()
    outs = []
    for n in ("boxes", "scores", "labels"):
        m = MagicMock(); m.name = n; outs.append(m)
    inp = MagicMock(); inp.name = "input"
    session.get_outputs.return_value = outs
    session.get_inputs.return_value  = [inp]
    session.run.return_value = [boxes, scores, labels]
    return session


@patch("ifnude.api.get_model")
def test_detect_returns_dicts(mock_get):
    mock_get.return_value = (
        _mock_session(
            np.array([[[[10,20,50,60]]]], dtype=np.float32),
            np.array([[[0.9]]], dtype=np.float32),
            np.array([[0]], dtype="int32"),
        ),
        ["EXPOSED_BREAST_F"],
    )
    from ifnude.api import detect
    result = detect(np.zeros((300,400,3), dtype=np.uint8))
    assert result and all("box" in r and "score" in r and "label" in r for r in result)


@patch("ifnude.api.get_model")
def test_detect_filters_low_score(mock_get):
    mock_get.return_value = (
        _mock_session(
            np.array([[[[10,10,50,50]]]], dtype=np.float32),
            np.array([[[0.2]]], dtype=np.float32),
            np.array([[0]], dtype="int32"),
        ),
        ["EXPOSED_BREAST_F"],
    )
    from ifnude.api import detect
    assert detect(np.zeros((300,400,3), dtype=np.uint8), min_prob=0.95) == []


def test_detect_invalid_mode():
    from ifnude.api import detect
    with pytest.raises(ValueError, match="mode"):
        detect(np.zeros((10,10,3), dtype=np.uint8), mode="turbo")


@patch("ifnude.api.get_model")
def test_detect_skips_out_of_range_label(mock_get):
    mock_get.return_value = (
        _mock_session(
            np.array([[[[10,10,50,50]]]], dtype=np.float32),
            np.array([[[0.9]]], dtype=np.float32),
            np.array([[99]], dtype="int32"),
        ),
        ["EXPOSED_BREAST_F"],
    )
    from ifnude.api import detect
    assert detect(np.zeros((300,400,3), dtype=np.uint8)) == []


def test_censor_raises_without_output():
    from ifnude.api import censor
    with pytest.raises(ValueError):
        censor(np.zeros((10,10,3), dtype=np.uint8))


@patch("ifnude.api.detect")
def test_censor_skips_detect_when_detections_supplied(mock_detect):
    from ifnude.api import censor
    img = np.zeros((100,100,3), dtype=np.uint8)
    censor(img, out_path="/tmp/out.jpg", detections=[])
    mock_detect.assert_not_called()


@patch("ifnude.api.get_model")
def test_detect_batch_returns_list_of_lists(mock_get):
    mock_get.return_value = (
        _mock_session(
            np.array([[[[10,20,50,60]]]], dtype=np.float32),
            np.array([[[0.9]]], dtype=np.float32),
            np.array([[0]], dtype="int32"),
        ),
        ["EXPOSED_BREAST_F"],
    )
    from ifnude.api import detect_batch
    imgs = [np.zeros((200,200,3), dtype=np.uint8) for _ in range(3)]
    results = detect_batch(imgs)
    assert len(results) == 3 and all(isinstance(r, list) for r in results)
