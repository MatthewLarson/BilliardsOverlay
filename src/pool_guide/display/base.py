"""Display sink contract: something that shows a rendered overlay image."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DisplaySink(ABC):
    @abstractmethod
    def show(self, overlay_bgr: np.ndarray) -> None:
        """Present one full-resolution overlay frame."""

    @abstractmethod
    def poll_key(self) -> int:
        """Return a pressed key code (-1 if none). Used to drive interactive apps."""

    def __enter__(self) -> "DisplaySink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:  # noqa: B027
        """Release windows/sockets. Override if needed."""
