"""Local display: an OpenCV window, optionally fullscreen on the projector.

Fullscreen placement across monitors is done by moving the window to an offset
before flagging it fullscreen. `monitor` is a 1-based index; monitor 1 is the
primary. If you can't get it on the projector automatically, drag it over once
and it will remember within the session.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import Config

WINDOW = "pool-guide"


class LocalDisplay:
    def __init__(self, cfg: Config, fullscreen: bool):
        self.w = cfg.display.width
        self.h = cfg.display.height
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if fullscreen:
            # Heuristic multi-monitor offset: assume projector sits to the right
            # of the primary display at that display's width * (monitor-1).
            offset_x = self.w * max(0, cfg.display.monitor - 1)
            cv2.moveWindow(WINDOW, offset_x, 0)
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW, self.w, self.h)

    def show(self, overlay_bgr: np.ndarray) -> None:
        if overlay_bgr.shape[1] != self.w or overlay_bgr.shape[0] != self.h:
            overlay_bgr = cv2.resize(overlay_bgr, (self.w, self.h))
        cv2.imshow(WINDOW, overlay_bgr)

    def poll_key(self) -> int:
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(WINDOW)
