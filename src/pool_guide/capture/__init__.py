"""Frame sources. `open_source(config)` returns the right one for the config."""
from __future__ import annotations

from ..config import Config
from .base import Frame, FrameSource


def open_source(cfg: Config) -> FrameSource:
    """Factory: pick a FrameSource based on capture.source (and network role)."""
    src = cfg.capture.source

    # In distributed/brain mode, frames always arrive over the network regardless
    # of what physical source the sensor node happens to use.
    if cfg.mode == "distributed" and cfg.network.role == "brain":
        src = "network"

    if src == "synthetic":
        from .synthetic import SyntheticSource
        return SyntheticSource(cfg)
    if src == "webcam":
        from .webcam import WebcamSource
        return WebcamSource(cfg)
    if src == "kinect_v1":
        from .kinect import KinectV1Source
        return KinectV1Source(cfg)
    if src == "network":
        from .network_client import NetworkFrameSource
        return NetworkFrameSource(cfg)
    raise ValueError(f"Unknown capture source: {src!r}")


__all__ = ["Frame", "FrameSource", "open_source"]
