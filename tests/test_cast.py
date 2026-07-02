"""Cast display sink: the MJPEG page server serves the latest overlay frame."""
import urllib.request

import cv2
import numpy as np

from pool_guide.config import Config
from pool_guide.display.cast import CastDisplay


def test_cast_serves_page_and_frame():
    cfg = Config()
    cfg.display.width, cfg.display.height = 320, 240
    cfg.display.cast_port = 0          # ephemeral
    cfg.display.cast_target = ""       # don't auto-cast
    disp = CastDisplay(cfg)
    try:
        port = disp._httpd.server_address[1]
        disp.show(np.full((240, 320, 3), 90, np.uint8))

        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
        assert "stream.mjpg" in page

        jpg = urllib.request.urlopen(f"http://127.0.0.1:{port}/frame.jpg", timeout=5).read()
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (240, 320, 3)
        assert disp.poll_key() == -1
    finally:
        disp.close()


def test_all_display_sinks_support_context_manager():
    # Regression: NetworkDisplay/LocalDisplay/CastDisplay must inherit DisplaySink
    # so `with open_sink(cfg) as sink:` works (calibration used it and crashed).
    from pool_guide.display.base import DisplaySink
    from pool_guide.display.local import LocalDisplay
    from pool_guide.display.network_sink import NetworkDisplay
    from pool_guide.display.cast import CastDisplay
    for cls in (LocalDisplay, NetworkDisplay, CastDisplay):
        assert issubclass(cls, DisplaySink)
        assert hasattr(cls, "__enter__") and hasattr(cls, "__exit__")
