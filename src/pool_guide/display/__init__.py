"""Display sinks. `open_sink(config)` returns the right one for the config."""
from __future__ import annotations

from ..config import Config
from .base import DisplaySink


def open_sink(cfg: Config) -> DisplaySink:
    sink = cfg.display.sink

    # In distributed/brain mode the overlay is streamed to the sensor node.
    if cfg.mode == "distributed" and cfg.network.role == "brain":
        sink = "network"

    if sink in ("window", "projector"):
        from .local import LocalDisplay
        return LocalDisplay(cfg, fullscreen=(sink == "projector"))
    if sink == "cast":
        from .cast import CastDisplay
        return CastDisplay(cfg)
    if sink == "network":
        from .network_sink import NetworkDisplay
        return NetworkDisplay(cfg)
    raise ValueError(f"Unknown display sink: {sink!r}")


__all__ = ["DisplaySink", "open_sink"]
