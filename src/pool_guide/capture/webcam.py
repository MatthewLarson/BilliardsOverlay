"""Webcam / UVC camera source via OpenCV. For dev without a Kinect."""
from __future__ import annotations

import cv2

from ..config import Config
from .base import Frame, FrameSource


class WebcamSource(FrameSource):
    def __init__(self, cfg: Config):
        self._flip = cfg.capture.flip_horizontal
        self._cap = cv2.VideoCapture(cfg.capture.webcam_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.capture.fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {cfg.capture.webcam_index}"
            )
        self._i = 0

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok:
            return None
        if self._flip:
            img = cv2.flip(img, 1)
        self._i += 1
        return Frame(rgb=img, depth=None, index=self._i)

    def close(self) -> None:
        self._cap.release()
