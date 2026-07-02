"""Frame source for the brain: pulls frames streamed by a remote sensor node."""
from __future__ import annotations

from ..config import Config
from ..net.protocol import Receiver
from .base import Frame, FrameSource


class NetworkFrameSource(FrameSource):
    def __init__(self, cfg: Config):
        # Brain binds the frame port; the sensor node connects and pushes to it.
        self._rx = Receiver(port=cfg.network.frame_port, host="*", bind=True)

    def read(self) -> Frame | None:
        got = self._rx.recv_image()
        if got is None:
            return None  # timeout -- caller can treat as "no frame yet"
        img, index, depth = got
        return Frame(rgb=img, depth=depth, index=index)

    def close(self) -> None:
        self._rx.close()
