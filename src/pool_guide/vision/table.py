"""Locate the playing surface in a camera frame.

Two ways to bound the felt, best first:

  1. If calibration has a camera->table homography, we know exactly where the
     rails are: warp the table-mm rectangle back into camera pixels. Precise and
     lighting-independent.

  2. Otherwise, find the felt by color. The felt is by far the largest coloured
     region, so the dominant hue in the frame IS the felt hue (works for green,
     blue, or red cloth). We mask that hue, take the biggest blob, and fill it.

Either way we erode inward a little so rails, cushions, and pocket jaws don't
leak into the ball search.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..calibration.model import Calibration, warp_points


def table_polygon_camera(calib: Calibration | None) -> np.ndarray | None:
    """Table corners in camera pixels from the table homography, or None."""
    if calib is None or calib.H_cam2table is None:
        return None
    L, W = calib.table_size_mm
    corners_mm = np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=np.float64)
    H_table2cam = np.linalg.inv(calib.H_cam2table)
    return warp_points(H_table2cam, corners_mm)


def table_quad_camera(mask: np.ndarray, calib: Calibration | None) -> np.ndarray | None:
    """Four table corners in camera px. Prefer the calibrated rectangle; else the
    min-area rectangle of the felt mask. Returns 4x2 (float) or None."""
    poly = table_polygon_camera(calib)
    if poly is not None:
        return poly
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(biggest))   # 4x2, ordered
    return box.astype(np.float64)


def px_per_mm(calib: Calibration | None, frame_shape) -> float | None:
    """Camera pixels per real millimetre, estimated from the table polygon."""
    poly = table_polygon_camera(calib)
    if poly is None:
        return None
    L, W = calib.table_size_mm
    top = np.linalg.norm(poly[1] - poly[0])       # px length of the long rail
    left = np.linalg.norm(poly[3] - poly[0])      # px length of the short rail
    scales = [s for s, mm in ((top, L), (left, W)) if mm > 0 for s in (s / mm,)]
    return float(np.mean(scales)) if scales else None


def _dominant_felt_hue(hsv: np.ndarray, min_s: int, min_v: int) -> int:
    """The felt colour = the dominant hue, weighted by saturation.

    Weighting hue votes by saturation lets a large but semi-desaturated (army
    green) felt beat the near-gray floor/rails, and works even when the felt is
    dark -- important on dim tables where absolute S/V thresholds would reject it.
    """
    h = hsv[..., 0]
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2]
    valid = (s >= min_s) & (v >= min_v)
    if int(np.count_nonzero(valid)) < 200:          # dim/desaturated: relax thresholds
        valid = (s >= max(8, min_s // 2)) & (v >= max(6, min_v // 2))
    if not valid.any():
        valid = np.ones(h.shape, dtype=bool)
    hist = np.zeros(180, np.float64)
    np.add.at(hist, h[valid].ravel().astype(np.intp), s[valid].ravel())
    # smooth the circular hue histogram so noise doesn't split the felt peak
    wrapped = np.concatenate([hist[-3:], hist, hist[:3]])
    hist = np.convolve(wrapped, np.ones(5) / 5, "same")[3:-3]
    return int(np.argmax(hist))


def build_table_mask(frame_bgr: np.ndarray, calib: Calibration | None,
                     cfg) -> tuple[np.ndarray, int | None]:
    """Return (uint8 mask of the play area, felt_hue or None).

    felt_hue is returned so the ball detector can reuse it to subtract the cloth.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    poly = table_polygon_camera(calib)
    if poly is not None:
        # Precise: fill the calibrated rectangle. Felt hue still sampled for the
        # detector, but only from inside the table.
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
        inside = mask > 0
        felt_hue = _dominant_felt_hue(hsv[inside].reshape(-1, 1, 3),
                                      cfg.min_saturation, cfg.min_value) \
            if inside.any() else None
    else:
        # Color-based: dominant hue is the felt; largest blob of it is the table.
        felt_hue = _dominant_felt_hue(hsv, cfg.min_saturation, cfg.min_value)
        tol = cfg.felt_hue_tolerance
        lo = np.array([max(0, felt_hue - tol), cfg.min_saturation, cfg.min_value])
        hi = np.array([min(179, felt_hue + tol), 255, 255])
        felt = cv2.inRange(hsv, lo, hi)
        felt = cv2.morphologyEx(felt, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        contours, _ = cv2.findContours(felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), np.uint8)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [biggest], -1, 255, cv2.FILLED)

    if cfg.table_erode_px > 0 and mask.any():
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.table_erode_px * 2 + 1,) * 2)
        mask = cv2.erode(mask, k)
    return mask, felt_hue
