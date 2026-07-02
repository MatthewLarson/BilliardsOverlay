"""Xbox 360 Kinect (v1) source via the `freenect` Python bindings for libfreenect.

Install (Linux / Raspberry Pi):
    sudo apt install freenect libfreenect-dev
    # build the Python wrapper from the libfreenect source tree (wrappers/python)
This is NOT pip-installable and does not work on Windows -- use `webcam` or
`synthetic` for dev on Windows, then switch to `kinect_v1` on the rig.

Notes from field reports:
  * Use a POWERED USB hub -- the Pi's ports can't feed the Kinect's camera/audio.
  * Pulling RGB + depth simultaneously can time out on the Pi; if you hit that,
    set capture.depth: false until Phase 2 needs it, or lower the frame rate.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from .base import Frame, FrameSource


class KinectV1Source(FrameSource):
    def __init__(self, cfg: Config):
        try:
            import freenect  # type: ignore
        except ImportError as e:  # pragma: no cover - hardware/platform specific
            raise RuntimeError(
                "The `freenect` module is not available. Build libfreenect with "
                "its Python wrapper on the sensor machine, or use capture.source "
                "'webcam'/'synthetic' for development. See src/pool_guide/capture/"
                "kinect.py for install notes."
            ) from e
        self._fn = freenect
        self._want_depth = cfg.capture.depth
        self._flip = cfg.capture.flip_horizontal
        self._i = 0

    def read(self) -> Frame | None:
        import cv2

        rgb, _ = self._fn.sync_get_video()  # HxWx3 RGB uint8
        if rgb is None:
            return None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self._flip:
            bgr = cv2.flip(bgr, 1)

        depth = None
        if self._want_depth:
            d, _ = self._fn.sync_get_depth()  # HxW uint11 (0..2047), mm-ish
            if d is not None:
                depth = np.asarray(d, dtype=np.uint16)
                if self._flip:
                    depth = cv2.flip(depth, 1)

        self._i += 1
        return Frame(rgb=bgr, depth=depth, index=self._i)

    def close(self) -> None:  # pragma: no cover
        try:
            self._fn.sync_stop()
        except Exception:
            pass
