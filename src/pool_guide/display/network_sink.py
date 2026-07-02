"""Network display sink for the brain: streams the overlay to the sensor node,
which paints it on the projector. Key events can't come back this way in Phase 0,
so poll_key() always returns -1 (the sensor node owns local keyboard input)."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..net.protocol import Sender


class NetworkDisplay:
    def __init__(self, cfg: Config):
        self._quality = cfg.network.jpeg_quality
        self._i = 0
        # Brain binds the overlay port; sensor node connects and pulls.
        self._tx = Sender(port=cfg.network.overlay_port, host="*", bind=True)

    def show(self, overlay_bgr: np.ndarray) -> None:
        self._i += 1
        self._tx.send_image(overlay_bgr, self._i, self._quality)

    def poll_key(self) -> int:
        return -1

    def close(self) -> None:
        self._tx.close()
