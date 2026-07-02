"""Chromecast display sink.

The projector is driven by a Chromecast (plugged into the projector's HDMI), not
a direct HDMI cable. We can't open a fullscreen window on it, so instead we:

  1. serve the live overlay as an MJPEG stream on this node, wrapped in a
     black fullscreen HTML page, and
  2. (optionally) tell the Chromecast to display that page URL using `catt`
     (Cast All The Things -- `pip install catt`).

The projector then shows the page. Because our ArUco calibration maps
camera->projector-pixels, the projector's physical position/skew (off to the
side, keystoned) is absorbed into the calibration automatically -- nothing here
needs to correct for it.

Note: casting adds latency (typically a few hundred ms). Calibration tolerates
it; live aim/prediction will feel a touch laggy over Chromecast. That's inherent
to casting, not this code.
"""
from __future__ import annotations

import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from ..config import Config
from .base import DisplaySink

_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Pool Guide overlay</title>
<style>html,body{{margin:0;height:100%;background:#000;overflow:hidden}}
img{{position:fixed;inset:0;width:100%;height:100%;object-fit:fill}}</style></head>
<body><img src="/stream.mjpg" alt=""></body></html>"""


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class CastDisplay(DisplaySink):
    def __init__(self, cfg: Config):
        self.w = cfg.display.width
        self.h = cfg.display.height
        self._fps = max(1, cfg.display.cast_fps)
        self._port = cfg.display.cast_port
        self._target = cfg.display.cast_target.strip()
        self._lock = threading.Lock()
        self._jpeg = self._encode(np.zeros((self.h, self.w, 3), np.uint8))

        disp = self
        self.url = f"http://{_lan_ip()}:{self._port}/"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/stream.mjpg":
                    return self._stream()
                if self.path in ("/frame.jpg",):
                    return self._one()
                body = _PAGE.format().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _one(self):
                data = disp.latest()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        data = disp.latest()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(data)).encode()
                                         + b"\r\n\r\n" + data + b"\r\n")
                        time.sleep(1.0 / disp._fps)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self._port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        print(f"[cast] overlay page at {self.url}")
        self._catt = None
        if self._target:
            self._start_cast()

    # -- overlay frame plumbing --
    def _encode(self, bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""

    def latest(self) -> bytes:
        with self._lock:
            return self._jpeg

    def show(self, overlay_bgr: np.ndarray) -> None:
        if overlay_bgr.shape[1] != self.w or overlay_bgr.shape[0] != self.h:
            overlay_bgr = cv2.resize(overlay_bgr, (self.w, self.h))
        data = self._encode(overlay_bgr)
        with self._lock:
            self._jpeg = data

    def poll_key(self) -> int:
        return -1

    # -- catt casting --
    def _start_cast(self):
        try:
            self._catt = subprocess.Popen(
                ["catt", "-d", self._target, "cast_site", self.url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[cast] casting {self.url} -> Chromecast '{self._target}'")
        except FileNotFoundError:
            print("[cast] `catt` not found -- install it (pip install catt) or cast "
                  f"the page manually: {self.url}")
        except Exception as e:                                    # noqa: BLE001
            print(f"[cast] could not start casting: {e}")

    def close(self) -> None:
        try:
            self._httpd.shutdown()
        except Exception:                                         # noqa: BLE001
            pass
        if self._catt and self._catt.poll() is None:
            self._catt.terminate()
