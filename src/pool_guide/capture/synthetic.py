"""Synthetic top-down pool table -- lets you develop with zero hardware.

Renders a green felt rectangle with rails, six pockets, and a few colored balls
so calibration/detection code has something plausible to chew on. The balls
drift slightly each frame so tracking code sees motion.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from .base import Frame, FrameSource


class SyntheticSource(FrameSource):
    def __init__(self, cfg: Config):
        self.w = cfg.capture.width
        self.h = cfg.capture.height
        self._i = 0
        # (x, y, radius, BGR color); positions are fractions of the felt area.
        self._balls = [
            [0.25, 0.50, 12, (255, 255, 255)],  # cue
            [0.55, 0.40, 12, (0, 0, 220)],      # red-ish
            [0.60, 0.55, 12, (0, 200, 220)],    # yellow-ish
            [0.70, 0.48, 12, (200, 120, 0)],    # blue-ish
        ]
        self.draw_cue = True
        # Ground truth exposed for tests: the unit aim direction (cue -> target)
        # and the cue ball's pixel centre in the most recently rendered frame.
        self.last_cue_dir: tuple[float, float] | None = None
        self.last_cue_ball_px: tuple[float, float] | None = None

    def read(self) -> Frame:
        import cv2

        img = np.zeros((self.h, self.w, 3), np.uint8)
        # Playing surface inset from the frame edges (simulates rails/cushions).
        m = int(0.10 * min(self.w, self.h))
        x0, y0, x1, y1 = m, m, self.w - m, self.h - m
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 110, 30), -1)   # felt
        cv2.rectangle(img, (x0, y0), (x1, y1), (30, 40, 90), max(2, m // 4))  # rail

        # Pockets: four corners + two side.
        pw, ph = x1 - x0, y1 - y0
        for px, py in [(x0, y0), (x1, y0), (x0, y1), (x1, y1),
                       (x0 + pw // 2, y0), (x0 + pw // 2, y1)]:
            cv2.circle(img, (px, py), max(6, m // 3), (0, 0, 0), -1)

        # Balls, drifting on a slow sinusoid so motion is visible.
        t = self._i / 30.0
        cue_px = None
        for j, b in enumerate(self._balls):
            fx = np.clip(b[0] + 0.02 * np.sin(t + b[1] * 6), 0.05, 0.95)
            fy = np.clip(b[1] + 0.02 * np.cos(t + b[0] * 6), 0.05, 0.95)
            cx = int(x0 + fx * pw)
            cy = int(y0 + fy * ph)
            if j == 0:
                cue_px = (cx, cy)
            cv2.circle(img, (cx, cy), b[2], b[3], -1)
            cv2.circle(img, (cx, cy), b[2], (20, 20, 20), 1)

        # Cue stick: a thick wood-coloured line pointing at the cue ball, slowly
        # sweeping so its angle changes frame to frame.
        if self.draw_cue and cue_px is not None:
            r = self._balls[0][2]
            theta = 0.6 * np.sin(t * 0.5) + 0.3     # aim angle, radians
            d = (float(np.cos(theta)), float(np.sin(theta)))  # cue -> ball -> beyond
            gap = r + 6
            tip = (cue_px[0] - d[0] * gap, cue_px[1] - d[1] * gap)
            length = 0.28 * float(np.hypot(self.w, self.h))
            butt = (tip[0] - d[0] * length, tip[1] - d[1] * length)
            cv2.line(img, (int(butt[0]), int(butt[1])), (int(tip[0]), int(tip[1])),
                     (150, 190, 225), 7)           # light tan shaft
            self.last_cue_dir = d
            self.last_cue_ball_px = (float(cue_px[0]), float(cue_px[1]))

        self._i += 1
        return Frame(rgb=img, depth=None, index=self._i)
