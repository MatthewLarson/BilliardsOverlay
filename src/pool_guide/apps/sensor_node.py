"""Distributed-mode agent for the Raspberry Pi.

Runs on the machine physically attached to the Kinect and projector. It:
  * captures frames locally and PUSHes them to the brain, and
  * PULLs the rendered overlay from the brain and paints it on the projector.

The brain (your PC) runs the real apps (calibrate / verify / later the live
predictor) with mode: distributed and network.role: brain.

Run on the Pi:
    python -m pool_guide.apps.sensor_node

Requires config.yaml with:
    mode: distributed
    network: { role: sensor, brain_host: <PC ip> }
    capture: { source: kinect_v1 }     # or webcam for testing
    display: { sink: projector }
"""
from __future__ import annotations

import argparse
import threading

import numpy as np

from ..capture import open_source
from ..config import load_config
from ..display.local import LocalDisplay
from ..net.protocol import Receiver, Sender


def _overlay_loop(rx: Receiver, display: LocalDisplay, stop: threading.Event):
    """Continuously pull overlays from the brain and show them."""
    blank = None
    while not stop.is_set():
        got = rx.recv_image()
        if got is None:
            continue
        img, _index, _depth = got
        display.show(img)
        if display.poll_key() in (27, ord("q")):
            stop.set()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pool Guide sensor node (Pi side)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.mode != "distributed" or cfg.network.role != "sensor":
        print("WARNING: sensor_node expects mode: distributed, network.role: sensor")

    src = open_source(cfg)                     # local Kinect/webcam
    display = LocalDisplay(cfg, fullscreen=(cfg.display.sink == "projector"))

    # Sensor connects out to the brain, which binds both ports.
    frame_tx = Sender(port=cfg.network.frame_port, host=cfg.network.brain_host, bind=False)
    overlay_rx = Receiver(port=cfg.network.overlay_port, host=cfg.network.brain_host,
                          bind=False, timeout_ms=1000)

    stop = threading.Event()
    t = threading.Thread(target=_overlay_loop, args=(overlay_rx, display, stop), daemon=True)
    t.start()

    print(f"Sensor node streaming to brain at {cfg.network.brain_host} "
          f"(frames:{cfg.network.frame_port}, overlay:{cfg.network.overlay_port}). "
          "Ctrl-C to stop.")
    try:
        for frame in src:
            if stop.is_set():
                break
            depth = frame.depth if isinstance(frame.depth, np.ndarray) else None
            frame_tx.send_image(frame.rgb, frame.index, cfg.network.jpeg_quality, depth)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        src.close()
        frame_tx.close()
        overlay_rx.close()
        display.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
