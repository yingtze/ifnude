"""ifnude.io.downloader — asset downloading with atomic writes and thread safety."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import httpx

from ..model.constants import MODEL_URL, CLASSES_URL, MODEL_PATH, CLASSES_PATH
from ..exceptions import AssetDownloadError

logger = logging.getLogger(__name__)

_ASSET_LOCK = threading.Lock()


def ensure_assets() -> None:
    """Download model and class list if not already cached (thread-safe)."""
    if MODEL_PATH.exists() and CLASSES_PATH.exists():
        return
    with _ASSET_LOCK:
        if not MODEL_PATH.exists():
            _download(MODEL_URL, MODEL_PATH)
        if not CLASSES_PATH.exists():
            _download(CLASSES_URL, CLASSES_PATH)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* atomically via a .tmp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading {dest.name}...")
    try:
        try:
            from tqdm import tqdm
            _stream(url, tmp, tqdm)
        except ImportError:
            _stream(url, tmp, None)
        tmp.rename(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise AssetDownloadError(f"Failed to download {url} → {dest}") from exc


def _stream(url: str, dest: Path, tqdm_cls) -> None:
    """Stream *url* into *dest*, with optional tqdm progress bar."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total      = int(r.headers.get("content-length", 0)) or None
        downloaded = 0
        bar        = tqdm_cls(total=total, desc=dest.name, unit="B",
                              unit_scale=True, unit_divisor=1024) if tqdm_cls else None

        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if bar:
                    bar.update(len(chunk))
                elif total:
                    print(f"\r  {downloaded*100//total}% ({downloaded}/{total} B)",
                          end="", flush=True)
                else:
                    print(f"\r  {downloaded} B downloaded", end="", flush=True)

        if bar:
            bar.close()
        else:
            print()
