"""Webcam / UVC camera source via OpenCV. For dev without a Kinect."""
from __future__ import annotations

import sys

import cv2

from ..config import Config
from .base import Frame, FrameSource


class WebcamSource(FrameSource):
    def __init__(self, cfg: Config):
        self._flip = cfg.capture.flip_horizontal
        idx = cfg.capture.webcam_index

        # On Linux (Raspberry Pi) the default backend is unreliable for UVC cams;
        # the explicit V4L2 backend is far more dependable. Fall back to default.
        self._cap = None
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                self._cap = cap
            else:
                cap.release()
        if self._cap is None:
            self._cap = cv2.VideoCapture(idx)

        # MJPG lets USB cams hit higher resolutions/fps than raw YUYV.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.capture.fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {idx}. Is another app (e.g. Cheese) "
                "using the camera?"
            )

        # Apply tuned V4L2 controls (brightness/contrast/saturation/gain/etc.)
        # AFTER opening, so they take effect on the live stream.
        if cfg.capture.controls:
            from . import camera_controls
            camera_controls.set_controls(
                camera_controls.device_path(idx), cfg.capture.controls)
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
