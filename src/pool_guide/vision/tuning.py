"""Score how well balls separate from the cloth in a frame -- drives autotune.

The felt is a semi-desaturated army green; balls are brighter/darker and more
colourful. A good image: felt is a well-exposed mid-tone with a stable hue, the
balls are detected and stand out from the felt in value and saturation, and
nothing is clipped to pure black/white.
"""
from __future__ import annotations

import cv2
import numpy as np

from .balls import BallDetector
from .table import build_table_mask


def score_separation(frame_bgr: np.ndarray, vision_cfg) -> dict:
    """Return {score, balls, contrast, felt_v, felt_s, clip}. Higher score = better."""
    h, w = frame_bgr.shape[:2]
    mask, felt_hue = build_table_mask(frame_bgr, None, vision_cfg)
    if felt_hue is None or int(np.count_nonzero(mask)) < 0.03 * h * w:
        return {"score": -1e6, "balls": 0, "contrast": 0.0,
                "felt_v": 0.0, "felt_s": 0.0, "clip": 1.0}

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    inside = mask > 0
    V = hsv[..., 2].astype(np.float32)
    S = hsv[..., 1].astype(np.float32)
    Vin, Sin = V[inside], S[inside]

    clip = float(np.mean(Vin < 6) + np.mean(Vin > 249))      # black/white clipping in table
    felt_v = float(np.median(Vin))
    felt_s = float(np.median(Sin))

    balls = BallDetector(vision_cfg, None).detect(frame_bgr, mask, felt_hue)
    n = len(balls)
    if balls:
        contrast = float(np.mean([abs(b.val - felt_v) + abs(b.sat - felt_s) for b in balls]))
    else:
        contrast = 0.0

    # Prefer the felt sitting in a comfortable mid-tone band (not too dark/bright).
    exposure_pen = abs(felt_v - 120.0) / 120.0

    score = (n * 6.0) + (contrast * 0.15) - (clip * 120.0) - (exposure_pen * 25.0)
    return {"score": float(score), "balls": n, "contrast": contrast,
            "felt_v": felt_v, "felt_s": felt_s, "clip": clip}
