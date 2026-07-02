"""Wire protocol for distributed mode, built on ZeroMQ.

Two independent channels:
  * frames:  sensor  --PUSH-->  brain   (JPEG-compressed camera frames, + depth)
  * overlay: brain   --PUSH-->  sensor  (JPEG-compressed rendered overlay)

We use PUSH/PULL so the receiver simply blocks for the next message and the
sender never accumulates unbounded backlog (ZMQ high-water mark drops old
frames, which is what you want for live video -- latency over completeness).

Each message is two ZMQ frames: a small JSON header, then the JPEG bytes.
Depth (uint16) is sent as a separate PNG-encoded message when present.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
import zmq


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_png16(depth: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", depth)  # lossless, preserves uint16
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def decode_png16(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


@dataclass
class Sender:
    """PUSH side. `bind=True` on the machine that owns the port."""
    port: int
    host: str = "*"
    bind: bool = True

    def __post_init__(self):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.SNDHWM, 2)     # drop stale frames under backpressure
        self._sock.setsockopt(zmq.LINGER, 0)
        addr = f"tcp://{self.host}:{self.port}"
        (self._sock.bind if self.bind else self._sock.connect)(addr)

    def send_image(self, img: np.ndarray, index: int, quality: int,
                   depth: np.ndarray | None = None) -> None:
        header = {"index": int(index), "has_depth": depth is not None}
        parts = [json.dumps(header).encode(), encode_jpeg(img, quality)]
        if depth is not None:
            parts.append(encode_png16(depth))
        self._sock.send_multipart(parts)

    def close(self):
        self._sock.close(0)


@dataclass
class Receiver:
    """PULL side. `bind=False` connects to a Sender that bound the port."""
    port: int
    host: str
    bind: bool = False
    timeout_ms: int = 2000

    def __post_init__(self):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.setsockopt(zmq.RCVHWM, 2)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        addr = f"tcp://{self.host}:{self.port}"
        (self._sock.bind if self.bind else self._sock.connect)(addr)

    def recv_image(self):
        """Return (img, index, depth|None), or None on timeout."""
        try:
            parts = self._sock.recv_multipart()
        except zmq.Again:
            return None
        header = json.loads(parts[0])
        img = decode_jpeg(parts[1])
        depth = decode_png16(parts[2]) if header.get("has_depth") and len(parts) > 2 else None
        return img, header["index"], depth

    def close(self):
        self._sock.close(0)
