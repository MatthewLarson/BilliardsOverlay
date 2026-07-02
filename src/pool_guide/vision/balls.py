"""Detect and roughly classify pool balls in a top-down camera frame.

Pipeline per frame:
  1. Restrict to the play area (table mask).
  2. Remove the felt colour -> what's left inside the table is balls + cue.
     (Optionally, background subtraction against an empty-table image instead.)
  3. Find contours, fit a circle to each, keep those that are round and
     ball-sized (expected radius comes from the table scale when calibrated).
  4. Sample each ball's mean colour and a white-fraction to guess a label:
     cue / 8-ball / solid / stripe, plus a coarse colour name.

Classification here is deliberately coarse -- reading the printed NUMBER on each
ball is a later refinement. Phase 1 only needs reliable positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..calibration.model import Calibration
from .table import px_per_mm

# Coarse hue -> colour name (OpenCV hue is 0..179).
_HUE_NAMES = [
    (0, "red"), (10, "orange"), (22, "yellow"), (35, "green"),
    (85, "cyan"), (100, "blue"), (130, "purple"), (160, "red"), (180, "red"),
]


def _hue_name(h: int) -> str:
    for edge, name in _HUE_NAMES:
        if h <= edge:
            return name
    return "red"


@dataclass
class Ball:
    cx: float
    cy: float
    radius: float
    color_bgr: tuple[int, int, int]
    hue: int
    sat: int
    val: int
    white_fraction: float
    label: str          # cue | 8 | solid | stripe
    color_name: str
    id: int = -1        # assigned by the tracker across frames

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)


class BallDetector:
    def __init__(self, cfg, calib: Calibration | None = None):
        self.cfg = cfg
        self.calib = calib
        self._radius_range = self._expected_radius_range(cfg, calib)

    def _expected_radius_range(self, cfg, calib) -> tuple[float, float]:
        """Prefer a physical estimate (ball mm * table scale); else config px."""
        scale = px_per_mm(calib, None) if calib is not None else None
        if scale:
            r = 0.5 * cfg.ball_diameter_mm * scale
            return (r * (1 - cfg.radius_tolerance), r * (1 + cfg.radius_tolerance))
        return (float(cfg.min_ball_radius_px), float(cfg.max_ball_radius_px))

    def detect(self, frame_bgr: np.ndarray, table_mask: np.ndarray,
               felt_hue: int | None,
               background_bgr: np.ndarray | None = None) -> list[Ball]:
        cfg = self.cfg
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        if (cfg.use_background_subtraction and background_bgr is not None):
            obj = self._foreground_by_background(frame_bgr, background_bgr)
        else:
            obj = self._foreground_by_felt(hsv, felt_hue)

        # Keep only what's on the table, and drop tiny speckle.
        obj = cv2.bitwise_and(obj, obj, mask=table_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, k)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rmin, rmax = self._radius_range
        min_area = np.pi * (rmin * 0.7) ** 2
        balls: list[Ball] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if not (rmin <= radius <= rmax):
                continue
            circularity = area / (np.pi * radius * radius + 1e-6)
            if circularity < cfg.min_circularity:
                continue
            balls.append(self._describe(frame_bgr, hsv, cx, cy, radius))
        return balls

    def _foreground_by_felt(self, hsv: np.ndarray, felt_hue: int | None) -> np.ndarray:
        """Anything sufficiently un-felt-coloured (and not near-black) is an object."""
        cfg = self.cfg
        if felt_hue is None:
            # No felt estimate: treat saturated OR bright pixels as candidates.
            s, v = hsv[..., 1], hsv[..., 2]
            return ((s > cfg.min_saturation) | (v > 200)).astype(np.uint8) * 255
        h = hsv[..., 0].astype(np.int16)
        dist = np.minimum(np.abs(h - felt_hue), 180 - np.abs(h - felt_hue))
        is_felt = (dist <= cfg.felt_hue_tolerance) & (hsv[..., 1] >= cfg.min_saturation)
        is_dark = hsv[..., 2] < cfg.min_value          # pockets / deep shadow
        obj = (~is_felt) & (~is_dark)
        return obj.astype(np.uint8) * 255

    def _foreground_by_background(self, frame, background) -> np.ndarray:
        diff = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return mask

    def _describe(self, bgr, hsv, cx, cy, radius) -> Ball:
        """Sample colour + white-fraction inside the ball and classify it."""
        h, w = bgr.shape[:2]
        sample = np.zeros((h, w), np.uint8)
        cv2.circle(sample, (int(cx), int(cy)), max(1, int(radius * 0.7)), 255, -1)
        pts = sample > 0
        mean_bgr = bgr[pts].mean(axis=0)
        mean_hsv = hsv[pts].mean(axis=0)
        hue, sat, val = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

        # Fraction of near-white pixels: high for the cue ball and for stripes.
        white = (hsv[..., 1] < 40) & (hsv[..., 2] > 180) & pts
        white_fraction = float(white.sum()) / float(max(1, pts.sum()))

        label = self._classify(hue, sat, val, white_fraction)
        return Ball(
            cx=float(cx), cy=float(cy), radius=float(radius),
            color_bgr=tuple(int(x) for x in mean_bgr),
            hue=hue, sat=sat, val=val, white_fraction=white_fraction,
            label=label, color_name=_hue_name(hue),
        )

    @staticmethod
    def _classify(hue, sat, val, white_fraction) -> str:
        if val < 60 and sat < 90:
            return "8"                      # very dark, low colour -> black 8-ball
        if sat < 60 and val > 170 and white_fraction > 0.5:
            return "cue"                     # bright, colourless, mostly white
        if 0.18 < white_fraction < 0.75:
            return "stripe"                  # significant white patch + colour
        return "solid"
