# ifnude — nudity detection that just works

A neural network–powered library that detects nudity in images of both real humans and drawings. Identifies exactly which NSFW body parts are visible, with optional black-box censoring.

<img src="https://i.imgur.com/0KPJbl9.jpg" width=600>

---

## Quickstart (no install required)

This project runs directly from source — no `pip install ifnude` needed.

**1. Install dependencies**

```bash
pip install opencv-python-headless pillow onnxruntime httpx numpy
```

> For GPU acceleration, use `onnxruntime-gpu` instead of `onnxruntime`.

**2. Clone and run**

```bash
git clone https://github.com/yingtze/ifnude
cd ifnude

python example.py image.jpg
python example.py image.jpg --mode fast
python example.py image.jpg --censor output.jpg --min-prob 0.7
```

**3. Import in your own scripts**

Scripts inside the repo folder work without any extra setup:

```python
from ifnude import detect, detect_batch, censor
```

If your script lives *outside* the repo, point Python to it first:

```python
import sys
sys.path.insert(0, '/home/yourname/projects/ifnude')  # path to the cloned repo
from ifnude import detect, detect_batch, censor
```

> **First run:** the detection model (~139 MB) downloads automatically to `~/.ifnude/` and is cached for all future calls.

---

## Usage

### `detect()` — single image

```python
from ifnude import detect

# Default mode — higher accuracy
results = detect('photo.jpg')

# Fast mode — ~3× faster, slightly lower accuracy
results = detect('photo.jpg', mode='fast')

# Custom confidence threshold (default: 0.6 / 0.5 in fast mode)
results = detect('photo.jpg', min_prob=0.75)
```

Accepts a file path (`str` / `Path`), a **PIL** `Image`, or a **cv2** BGR numpy array.

**Example output:**

```python
[
    {
        'box': [164, 188, 246, 271],
        'score': 0.8253238201141357,
        'label': 'EXPOSED_BREAST_F'
    },
    {
        'box': [252, 190, 335, 270],
        'score': 0.8235630989074707,
        'label': 'EXPOSED_BREAST_F'
    }
]
```

Each detection contains:
- `box` — `[x1, y1, x2, y2]` bounding box in pixels
- `score` — confidence between 0.0 and 1.0
- `label` — one of the detectable label strings (see table below)

An empty list `[]` means no detections above the confidence threshold.

---

### `detect_batch()` — multiple images

```python
from ifnude import detect_batch

paths = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
results = detect_batch(paths, mode='fast')

# results is a list of lists — one per input image
for path, detections in zip(paths, results):
    print(f'{path}: {len(detections)} detection(s)')
```

**Example output:**

```python
[
    [{'box': [164, 188, 246, 271], 'score': 0.825, 'label': 'EXPOSED_BREAST_F'}],
    [],
    [{'box': [80, 100, 180, 200], 'score': 0.761, 'label': 'EXPOSED_GENITALIA_F'}]
]
```

The ONNX model loads once regardless of list length — significantly faster than looping `detect()` manually.

---

### `censor()` — black-box detected regions

```python
from ifnude import censor

# Save censored image to disk
censor('input.jpg', out_path='output.jpg')

# Censor only specific labels
censor('input.jpg', out_path='output.jpg', parts_to_blur=['EXPOSED_BREAST_F'])

# Pass pre-computed detections to avoid running inference twice
detections = detect('input.jpg')
censor('input.jpg', out_path='output.jpg', detections=detections)

# Display result (requires an active display / GUI environment)
censor('input.jpg', visualize=True)
```

Returns the censored image as a BGR numpy array regardless of whether `out_path` is set.

---

### `example.py` — CLI

```bash
# Detect and print results
python example.py photo.jpg

# Fast mode
python example.py photo.jpg --mode fast

# Save censored copy
python example.py photo.jpg --censor output.jpg

# Raise confidence threshold
python example.py photo.jpg --min-prob 0.75
```

**Example terminal output:**

```
EXPOSED_BREAST_F               score=0.825  box=[164, 188, 246, 271]
EXPOSED_BREAST_F               score=0.823  box=[252, 190, 335, 270]
```

---

## Detectable labels

| Label | Description |
|---|---|
| `EXPOSED_BREAST_F` | Exposed female breast |
| `EXPOSED_BREAST_M` | Exposed male breast |
| `EXPOSED_GENITALIA_F` | Exposed female genitalia |
| `EXPOSED_GENITALIA_M` | Exposed male genitalia |
| `EXPOSED_ANUS` | Exposed anus |
| `COVERED_BREAST_F` | Covered female breast |
| `COVERED_GENITALIA_F` | Covered female genitalia |
| `COVERED_GENITALIA_M` | Covered male genitalia |
| `COVERED_BUTTOCKS` | Covered buttocks |
| `EXPOSED_BUTTOCKS` | Exposed buttocks |

---

## Project structure

```
ifnude/
├── api.py               # detect(), detect_batch(), censor()
├── types.py             # ImageInput, Detection type aliases
├── exceptions.py        # IfnudeError, AssetDownloadError, ModelLoadError, InvalidOutputError
├── model/
│   ├── constants.py     # URLs, paths, label config
│   ├── loader.py        # ONNX session singleton (lru_cache)
│   └── resolver.py      # output tensor name resolution
└── io/
    ├── downloader.py    # atomic asset download, thread-safe
    └── image.py         # preprocessing, resize, read utilities

tests/
├── test_api.py
├── test_image.py
├── test_model_loader.py
└── test_downloader.py

example.py               # CLI entry point
```

---

## Requirements

- Python ≥ 3.9
- `opencv-python-headless` `pillow` `onnxruntime` `httpx` `numpy`

---

## Credits

Forked from [s0md3v/ifnude](https://github.com/s0md3v/ifnude), itself a fork of [NudeNet](https://pypi.org/project/NudeNet/) (no longer maintained).  
Detection model hosted on [HuggingFace](https://huggingface.co/s0md3v/nudity-checker).  
This fork: [github.com/yingtze/ifnude](https://github.com/yingtze/ifnude) — See [CHANGELOG.md](CHANGELOG.md) for full version history.
