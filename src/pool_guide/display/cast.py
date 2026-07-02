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

import os
import socket
import subprocess
import sys
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
        self._clients = 0                 # active MJPEG viewers (the Chromecast)
        self._clients_lock = threading.Lock()
        self._last_cast = 0.0
        self._casting = False

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
                with disp._clients_lock:
                    disp._clients += 1
                try:
                    while True:
                        data = disp.latest()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(data)).encode()
                                         + b"\r\n\r\n" + data + b"\r\n")
                        time.sleep(1.0 / disp._fps)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                finally:
                    with disp._clients_lock:
                        disp._clients -= 1

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self._port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        print(f"[cast] overlay page at {self.url}")
        self._catt = None
        self._stop_watch = threading.Event()
        if self._target:
            self._start_cast()
            threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        """Chromecast/DashCast sessions drop (timeouts, sensor restarts). When no
        client is pulling the stream, the Chromecast isn't showing our page -- so
        re-cast. Throttled so a just-started cast has time to connect."""
        while not self._stop_watch.wait(8.0):
            with self._clients_lock:
                n = self._clients
            if n <= 0 and not self._casting and (time.time() - self._last_cast) > 15:
                print("[cast] Chromecast not connected -- re-casting")
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
    @staticmethod
    def _catt_cmd() -> str:
        """Resolve catt next to this interpreter (the venv), since a systemd
        service's PATH usually doesn't include the venv's bin directory."""
        bindir = os.path.dirname(sys.executable)
        for name in ("catt", "catt.exe"):
            p = os.path.join(bindir, name)
            if os.path.exists(p):
                return p
        return "catt"

    def _start_cast(self):
        """(Re)cast in a background thread. Always `stop` first: cast_site HANGS if
        a previous DashCast session is wedged, but stop-then-cast reconnects."""
        if self._casting:
            return
        self._casting = True
        cmd = self._catt_cmd()

        def worker():
            def run(args, t):
                try:
                    subprocess.run([cmd, "-d", self._target, *args], timeout=t,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except FileNotFoundError:
                    print(f"[cast] catt not found at {cmd} -- pip install catt, or "
                          f"open {self.url} on the projector manually.")
                    return None
                except Exception:                                 # noqa: BLE001 (timeout etc.)
                    return False
            print(f"[cast] casting {self.url} -> '{self._target}'")
            if run(["stop"], 15) is not None:
                run(["cast_site", self.url], 40)
            self._last_cast = time.time()
            self._casting = False

        threading.Thread(target=worker, daemon=True).start()

    def close(self) -> None:
        self._stop_watch.set()
        try:
            self._httpd.shutdown()
        except Exception:                                         # noqa: BLE001
            pass
        if self._catt and self._catt.poll() is None:
            self._catt.terminate()
