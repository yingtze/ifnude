"""Tests for ifnude.io.downloader."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestEnsureAssets:
    def test_skips_download_when_both_exist(self, tmp_path):
        model = tmp_path / "detector.onnx"
        classes = tmp_path / "classes"
        model.write_bytes(b"fake")
        classes.write_text("EXPOSED_BREAST_F")
        with patch("ifnude.io.downloader.MODEL_PATH", model), \
             patch("ifnude.io.downloader.CLASSES_PATH", classes), \
             patch("ifnude.io.downloader._download") as mock_dl:
            from ifnude.io.downloader import ensure_assets
            ensure_assets()
            mock_dl.assert_not_called()

    def test_cleans_tmp_on_download_failure(self, tmp_path):
        from ifnude.exceptions import AssetDownloadError
        dest = tmp_path / "detector.onnx"
        with patch("ifnude.io.downloader.MODEL_PATH", dest), \
             patch("ifnude.io.downloader.CLASSES_PATH", tmp_path / "classes"), \
             patch("ifnude.io.downloader._stream", side_effect=RuntimeError("net error")):
            from ifnude.io import downloader
            with pytest.raises(AssetDownloadError):
                downloader._download("http://example.com/model", dest)
            assert not dest.with_suffix(".onnx.tmp").exists()
