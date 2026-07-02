"""Phase 2 entrypoint: detect the cue, project the aim line + ghost ball.

Run:
    python -m pool_guide.apps.aim_assist            # projects onto the table
    python -m pool_guide.apps.aim_assist --debug    # + camera-view window

Detects the cue ball and cue stick, casts the aim ray (bouncing off cushions up
to vision.aim_max_bounces), and projects:
  * the cue-ball path (white, dashed at bounces),
  * a "ghost ball" ring at first ball contact, and
  * the struck ball's predicted direction (yellow).

Geometry only for now -- strength and english arrive with the physics engine in
Phase 3. Requires calibration.json to project; without it, falls back to a
camera-view window.
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
from ..vision import (
    BallDetector,
    CueDetector,
    SimpleTracker,
    build_table_mask,
    compute_aim,
    table_quad_camera,
)

WHITE = (255, 255, 255)
YELLOW = (0, 215, 255)
CYAN = (255, 255, 0)


def _pick_cue_ball(balls):
    cues = [b for b in balls if b.label == "cue"]
    if cues:
        return max(cues, key=lambda b: b.white_fraction)
    # fall back to the whitest ball
    return max(balls, key=lambda b: b.white_fraction) if balls else None


def _draw_aim_camera(canvas, aim, radius):
    pts = [np.array(p) for p in aim.path]
    for i in range(len(pts) - 1):
        cv2.line(canvas, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), WHITE, 2)
    for rp in aim.rail_points:
        cv2.circle(canvas, tuple(np.array(rp).astype(int)), 5, CYAN, -1)
    if aim.contact == "ball" and aim.ghost_center is not None:
        gc = np.array(aim.ghost_center).astype(int)
        cv2.circle(canvas, tuple(gc), int(radius), YELLOW, 2)
        if aim.object_dir is not None and aim.object_center is not None:
            oc = np.array(aim.object_center)
            end = oc + np.array(aim.object_dir) * radius * 6
            cv2.line(canvas, tuple(oc.astype(int)), tuple(end.astype(int)), YELLOW, 2)


def _draw_aim_projector(overlay, calib, aim, radius):
    def W(pts):
        return warp_points(calib.H_cam2proj, pts)

    if len(aim.path) >= 2:
        pp = W(aim.path).astype(int)
        for i in range(len(pp) - 1):
            cv2.line(overlay, tuple(pp[i]), tuple(pp[i + 1]), WHITE, 2)
    for rp in aim.rail_points:
        p = W([rp])[0].astype(int)
        cv2.circle(overlay, tuple(p), 6, CYAN, -1)
    if aim.contact == "ball" and aim.ghost_center is not None:
        gc = W([aim.ghost_center])[0].astype(int)
        # project the radius by warping an edge point
        redge = W([aim.ghost_center, (aim.ghost_center[0] + radius, aim.ghost_center[1])])
        rproj = max(4, int(np.linalg.norm(redge[1] - redge[0])))
        cv2.circle(overlay, tuple(gc), rproj, YELLOW, 2)
        if aim.object_dir is not None and aim.object_center is not None:
            oc = np.array(aim.object_center)
            end = oc + np.array(aim.object_dir) * radius * 6
            seg = W([tuple(oc), tuple(end)]).astype(int)
            cv2.line(overlay, tuple(seg[0]), tuple(seg[1]), YELLOW, 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cue tracking + projected aim line")
    ap.add_argument("--config", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = None
    if os.path.exists(cfg.calibration.path):
        try:
            calib = load_calibration(cfg.calibration.path)
        except Exception as e:
            print(f"WARNING: calibration load failed ({e}); camera-view only.")
    else:
        print("No calibration.json -- showing aim in a camera window "
              "(run apps.calibrate to project onto the table).")

    detector = BallDetector(cfg.vision, calib)
    cue_detector = CueDetector(cfg.vision)
    tracker = SimpleTracker(cfg.vision.tracker_max_dist_px)
    proj_w, proj_h = (calib.projector_size if calib else (cfg.display.width, cfg.display.height))

    with open_source(cfg) as src, open_sink(cfg) as sink:
        print("Aim assist running. Press q/Esc to quit.")
        while True:
            frame = src.read()
            if frame is None:
                if sink.poll_key() in (27, ord("q")):
                    break
                continue

            mask, felt_hue = build_table_mask(frame.rgb, calib, cfg.vision)
            balls = tracker.update(detector.detect(frame.rgb, mask, felt_hue))
            cue_ball = _pick_cue_ball(balls)
            quad = table_quad_camera(mask, calib)

            aim = None
            cue = cue_detector.detect(frame.rgb, cue_ball.center if cue_ball else None)
            if cue is not None and cue_ball is not None and quad is not None:
                others = [(b.center, b.radius) for b in balls if b is not cue_ball]
                aim = compute_aim(
                    cue_ball.center, cue.direction, others, quad, cue_ball.radius,
                    max_bounces=cfg.vision.aim_max_bounces,
                    show_object_dir=cfg.vision.aim_show_object_dir,
                )

            if calib is not None:
                overlay = np.zeros((proj_h, proj_w, 3), np.uint8)
                if aim is not None:
                    _draw_aim_projector(overlay, calib, aim, cue_ball.radius)
                sink.show(overlay)
                if args.debug:
                    cam = frame.rgb.copy()
                    if aim is not None:
                        _draw_aim_camera(cam, aim, cue_ball.radius)
                    cv2.imshow("camera (debug)", cam)
            else:
                cam = frame.rgb.copy()
                if aim is not None:
                    _draw_aim_camera(cam, aim, cue_ball.radius)
                sink.show(cam)

            key = sink.poll_key()
            if args.debug:
                key = (cv2.waitKey(1) & 0xFF) if key == -1 else key
            if key in (27, ord("q")):
                break
    if args.debug:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
