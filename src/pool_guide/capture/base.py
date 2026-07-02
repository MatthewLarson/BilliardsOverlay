"""Frame source contract shared by every capture backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A single captured frame.

    rgb:   HxWx3 uint8 BGR image (OpenCV convention).
    depth: optional HxW uint16 depth map in millimetres (Kinect only), else None.
    index: monotonically increasing frame counter.
    """
    rgb: np.ndarray
    depth: np.ndarray | None = None
    index: int = 0


class FrameSource(ABC):
    """Anything that yields Frames: a Kinect, a webcam, a network stream, a fake."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next Frame, or None if the stream has ended."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:  # noqa: B027  (intentional no-op default)
        """Release any hardware/socket handles. Override if needed."""

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
