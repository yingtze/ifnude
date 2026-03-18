"""
example.py — jalankan langsung tanpa pip install ifnude

Usage:
    python example.py /path/to/image.jpg
    python example.py /path/to/image.jpg --mode fast
    python example.py /path/to/image.jpg --censor output.jpg
"""

import sys
import argparse
from pathlib import Path

# Tambahkan root project ke path agar `ifnude` bisa diimport langsung
sys.path.insert(0, str(Path(__file__).parent))

from ifnude import detect, censor


def main():
    parser = argparse.ArgumentParser(description="ifnude — nudity detection")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--mode", choices=["default", "fast"], default="default",
                        help="Detection mode (default: default)")
    parser.add_argument("--min-prob", type=float, default=None,
                        help="Minimum confidence threshold")
    parser.add_argument("--censor", metavar="OUT_PATH",
                        help="Save censored image to this path")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: file tidak ditemukan: {args.image}")
        sys.exit(1)

    results = detect(args.image, mode=args.mode, min_prob=args.min_prob)

    if not results:
        print("Tidak ada deteksi.")
    else:
        for r in results:
            print(f"  {r['label']:<30} score={r['score']:.3f}  box={r['box']}")

    if args.censor:
        # Pass pre-computed detections — avoids running inference a second time
        censor(args.image, out_path=args.censor, detections=results)
        print(f"\nCensored image disimpan ke: {args.censor}")


if __name__ == "__main__":
    main()
