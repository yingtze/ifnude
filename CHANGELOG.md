# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-03-18

This release is a full modernisation of the original fork. The public API (`detect`, `detect_batch`, `censor`) is unchanged and backward-compatible.

### Added

- **`detect_batch(images, mode, min_prob)`** — run detection across a list of images while loading the ONNX session only once. Faster than calling `detect()` in a loop when processing multiple files.
- **`detections=` parameter on `censor()`** — pass pre-computed results from `detect()` to skip running inference a second time.
- **GPU auto-detection** — `CUDAExecutionProvider` is selected automatically when available; falls back to CPU without any configuration.
- **Custom exception hierarchy** — `IfnudeError`, `AssetDownloadError`, `ModelLoadError`, `InvalidOutputError` allow callers to handle failures precisely.
- **`__version__`** exported from the package root.
- **`tests/`** — 21 unit tests across four focused test modules, none of which require a model download.
- **`example.py`** — zero-install CLI entry point; runs directly from source.

### Changed

- **Package structure** — monolithic `detector.py` (301 lines, 4 concerns) refactored into a clean module hierarchy:
  - `api.py` — public functions
  - `model/` — `constants.py`, `loader.py`, `resolver.py`
  - `io/` — `downloader.py`, `image.py`
  - `types.py`, `exceptions.py`
- **Model loading** — ONNX session is now a lazy singleton via `functools.lru_cache`. Previously the model was reloaded on every `detect()` call, wasting ~500 ms and memory each time.
- **Asset download** — replaced raw `urllib.request` with `httpx` streaming. Downloads are now atomic (written to `.tmp`, renamed on success) so a failed download never leaves a corrupt file on disk.
- **Thread safety** — `threading.Lock` + double-checked locking prevents concurrent threads from downloading the same asset simultaneously.
- **ONNX output resolution** — outputs are now resolved by tensor name (`boxes`, `scores`, `labels`) rather than fragile `isinstance` type checks. Falls back gracefully with a logged warning if names differ.
- **`censor()`** — accepts numpy arrays directly (not just file paths); raises `ValueError` when neither `out_path` nor `visualize=True` is provided (previously printed a warning and silently returned).
- **`detect()`** — validates `mode` argument; raises `ValueError` for unrecognised values instead of silently falling through to `else`.
- **Image preprocessing** — removed unnecessary round-trip through PIL for numpy array inputs; input is now used directly as `float32`.
- **`pyproject.toml`** (PEP 517/518) replaces legacy `setup.py`. Optional dependency groups: `[gpu]` for `onnxruntime-gpu`, `[dev]` for test tooling.
- All public functions carry complete type annotations.

### Fixed

- `raise NotImplemented` → `raise NotImplementedError` in preprocessing utilities. The original raised `TypeError` at runtime instead of the intended error.
- `label_idx` out-of-bounds access — index is now bounds-checked before accessing `classes[label_idx]`; out-of-range values are skipped with a warning instead of crashing with `IndexError`.
- `content-length` missing from download response — `total` is now `None` (not `0`) when the header is absent; progress display degrades to a bytes counter instead of showing `0%`.

### Removed

- `image_utils.py` — dead code inherited from the NudeNet fork; no function in this file was called anywhere in the codebase.
- `setup.py` — superseded by `pyproject.toml`.

### Deprecated

- `ifnude.detector` — use `from ifnude import detect` instead. The module remains as a shim that re-exports the public API and emits `DeprecationWarning`.
- `ifnude.detector_utils` — use `ifnude.io.image` instead. Same shim behaviour.

---

## [0.0.3] — prior release (s0md3v/ifnude)

- Initial fork of [s0md3v/ifnude](https://github.com/s0md3v/ifnude), itself a fork of [NudeNet](https://pypi.org/project/NudeNet/) with video detection removed due to crash-prone behaviour.
- ONNX-based detection model hosted on HuggingFace.
- `detect()` and `censor()` public API established.
