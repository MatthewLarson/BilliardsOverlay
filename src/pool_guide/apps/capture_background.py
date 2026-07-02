"""Capture an empty-table reference image for background-subtraction detection.

Clear all balls off the table, then run:
    python -m pool_guide.apps.capture_background

Averages a burst of frames (reduces sensor noise) and writes it to
vision.background_path (default background.jpg). Enable it later with
vision.use_background_subtraction: true.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..capture import open_source
from ..config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Capture empty-table background")
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print(f"Averaging {args.frames} frames of the EMPTY table...")
    acc = None
    n = 0
    with open_source(cfg) as src:
        for _ in range(args.frames * 3):     # allow for dropped/None frames
            frame = src.read()
            if frame is None:
                continue
            f = frame.rgb.astype(np.float32)
            acc = f if acc is None else acc + f
            n += 1
            if n >= args.frames:
                break
    if acc is None or n == 0:
        print("ERROR: captured no frames.")
        return 1
    background = (acc / n).astype(np.uint8)
    cv2.imwrite(cfg.vision.background_path, background)
    print(f"Saved {cfg.vision.background_path} from {n} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
