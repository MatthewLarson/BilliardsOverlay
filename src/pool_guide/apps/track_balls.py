"""Phase 1 entrypoint: detect balls and project a dot + label onto each.

Run:
    python -m pool_guide.apps.track_balls            # projects onto the table
    python -m pool_guide.apps.track_balls --debug    # also shows a camera-view window

If calibration.json exists, ball positions are mapped camera->projector so the
dots land on the real balls -- this is the visual proof that detection and
calibration agree. Without calibration it falls back to drawing detections on
the camera image (no projection).
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from ..calibration import load_calibration
from ..calibration.model import warp_points
from ..capture import open_source
from ..config import load_config
from ..display import open_sink
from ..vision import BallDetector, SimpleTracker, build_table_mask

_LABEL_BGR = {"cue": (255, 255, 255), "8": (60, 60, 60),
              "stripe": (0, 215, 255), "solid": (0, 200, 0)}


def _load_calib_if_any(path):
    if os.path.exists(path):
        try:
            return load_calibration(path)
        except Exception as e:
            print(f"WARNING: could not load calibration ({e}); running in debug view.")
    else:
        print("No calibration.json -- drawing detections on the camera image "
              "(run apps.calibrate to project onto the table).")
    return None


def _project_radius(H, cx, cy, r):
    """Map a ball's radius into projector pixels by warping an edge point."""
    p = warp_points(H, [[cx, cy], [cx + r, cy]])
    return max(3, int(np.linalg.norm(p[1] - p[0])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Detect balls and project dots")
    ap.add_argument("--config", default=None)
    ap.add_argument("--debug", action="store_true", help="show camera-view detections")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = _load_calib_if_any(cfg.calibration.path)
    detector = BallDetector(cfg.vision, calib)
    tracker = SimpleTracker(cfg.vision.tracker_max_dist_px)

    background = None
    if cfg.vision.use_background_subtraction and os.path.exists(cfg.vision.background_path):
        background = cv2.imread(cfg.vision.background_path)

    proj_w, proj_h = (calib.projector_size if calib else (cfg.display.width, cfg.display.height))

    with open_source(cfg) as src, open_sink(cfg) as sink:
        print("Tracking balls. Press q/Esc to quit.")
        while True:
            frame = src.read()
            if frame is None:
                if sink.poll_key() in (27, ord("q")):
                    break
                continue

            table_mask, felt_hue = build_table_mask(frame.rgb, calib, cfg.vision)
            balls = detector.detect(frame.rgb, table_mask, felt_hue, background)
            balls = tracker.update(balls)

            if calib is not None:
                overlay = _render_projector(calib, balls, proj_w, proj_h)
                sink.show(overlay)
            else:
                sink.show(_render_camera(frame.rgb, balls))

            if args.debug and calib is not None:
                cv2.imshow("camera (debug)", _render_camera(frame.rgb, balls))

            key = sink.poll_key()
            if args.debug:
                key = (cv2.waitKey(1) & 0xFF) if key == -1 else key
            if key in (27, ord("q")):
                break
    if args.debug:
        cv2.destroyAllWindows()
    return 0


def _render_projector(calib, balls, w, h) -> np.ndarray:
    overlay = np.zeros((h, w, 3), np.uint8)
    if not balls:
        return overlay
    centers = warp_points(calib.H_cam2proj, [b.center for b in balls])
    for b, (px, py) in zip(balls, centers):
        color = _LABEL_BGR.get(b.label, (0, 200, 0))
        r = _project_radius(calib.H_cam2proj, b.cx, b.cy, b.radius)
        cv2.circle(overlay, (int(px), int(py)), r, color, 2)
        cv2.putText(overlay, f"{b.id}:{b.label}", (int(px) + r + 4, int(py)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def _render_camera(frame, balls) -> np.ndarray:
    canvas = frame.copy()
    for b in balls:
        color = _LABEL_BGR.get(b.label, (0, 200, 0))
        cv2.circle(canvas, (int(b.cx), int(b.cy)), int(b.radius), color, 2)
        cv2.putText(canvas, f"{b.id}:{b.label}:{b.color_name}",
                    (int(b.cx) + int(b.radius) + 3, int(b.cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(canvas, f"balls: {len(balls)}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
