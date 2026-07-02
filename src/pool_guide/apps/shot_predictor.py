"""Phase 3 entrypoint: projected shot prediction with strength + english controls.

Run:
    python -m pool_guide.apps.shot_predictor            # projects onto the table
    python -m pool_guide.apps.shot_predictor --debug    # + camera-view window

Detects the balls and cue, reads the projected strength meter and cue-ball
contact point, runs the physics engine, and projects the true predicted paths of
every ball (with cushions, collisions, spin, and potting). Adjust while aiming:

    [ ]  strength      a d  side english      w s  follow/draw      c  centre
    q/Esc quit

Requires a FULL calibration (including the table corners, i.e. the camera->table
homography) so ball pixels can be turned into real millimetres for the physics.
Re-run `python -m pool_guide.apps.calibrate` and click the four corners if the
camera->table mapping is missing.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from ..calibration import load_calibration
from ..calibration.model import warp_points
from ..capture import open_source
from ..config import load_config
from ..display import open_sink
from ..physics import Shot, make_engine
from ..ui import ShotControls, draw_controls
from ..vision import (
    BallDetector,
    CueDetector,
    SimpleTracker,
    build_table_mask,
)
from .aim_assist import _pick_cue_ball

WHITE = (255, 255, 255)
YELLOW = (0, 215, 255)
RED = (60, 60, 235)


def _require_calibration(cfg):
    if not os.path.exists(cfg.calibration.path):
        print("ERROR: no calibration.json. Run `python -m pool_guide.apps.calibrate` "
              "first (and click the four table corners).", file=sys.stderr)
        return None
    calib = load_calibration(cfg.calibration.path)
    if calib.H_cam2table is None:
        print("ERROR: calibration has no camera->table homography. Re-run "
              "`python -m pool_guide.apps.calibrate` WITHOUT --skip-table and click "
              "the four table corners.", file=sys.stderr)
        return None
    return calib


def _draw_paths(canvas, traj, warp, id_colors, radius_proj_fn):
    """Draw every ball's predicted path. `warp` maps table-mm points to canvas px."""
    for bid, pts in traj.paths.items():
        if len(pts) < 2:
            continue
        color = WHITE if bid == "cue" else id_colors.get(bid, (0, 200, 0))
        if bid in traj.potted:
            color = RED
        pp = warp(pts).astype(np.int32)
        cv2.polylines(canvas, [pp], False, color, 2)
    if traj.ghost is not None:
        g = warp([traj.ghost])[0].astype(int)
        cv2.circle(canvas, tuple(g), radius_proj_fn(traj.ghost), YELLOW, 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Projected shot prediction (Phase 3)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = _require_calibration(cfg)
    if calib is None:
        return 2

    H_table2cam = np.linalg.inv(calib.H_cam2table)
    H_table2proj = calib.H_cam2proj @ H_table2cam
    proj_w, proj_h = calib.projector_size

    detector = BallDetector(cfg.vision, calib)
    cue_detector = CueDetector(cfg.vision)
    tracker = SimpleTracker(cfg.vision.tracker_max_dist_px)
    controls = ShotControls.from_config(cfg.controls)
    engine = make_engine(cfg.physics)

    r_mm = cfg.physics.ball_diameter_mm / 2
    L, W = calib.table_size_mm
    table = (0.0, 0.0, float(L), float(W))

    def to_proj(mm_pts):
        return warp_points(H_table2proj, mm_pts)

    def to_cam(mm_pts):
        return warp_points(H_table2cam, mm_pts)

    def radius_proj(mm_pt):
        e = to_proj([mm_pt, (mm_pt[0] + r_mm, mm_pt[1])])
        return max(4, int(np.linalg.norm(e[1] - e[0])))

    def radius_cam(mm_pt):
        e = to_cam([mm_pt, (mm_pt[0] + r_mm, mm_pt[1])])
        return max(3, int(np.linalg.norm(e[1] - e[0])))

    with open_source(cfg) as src, open_sink(cfg) as sink:
        print(f"Shot predictor running ({cfg.physics.engine} engine). "
              "[ ] strength  a d side  w s follow/draw  c centre  q quit")
        while True:
            frame = src.read()
            if frame is None:
                if sink.poll_key() in (27, ord("q")):
                    break
                continue

            mask, felt_hue = build_table_mask(frame.rgb, calib, cfg.vision)
            balls = tracker.update(detector.detect(frame.rgb, mask, felt_hue))
            cue_ball = _pick_cue_ball(balls)
            cue = cue_detector.detect(frame.rgb, cue_ball.center if cue_ball else None)

            traj = None
            id_colors: dict[str, tuple] = {}
            if cue is not None and cue_ball is not None:
                # Map everything into table millimetres.
                cue_mm = tuple(warp_points(calib.H_cam2table, [cue_ball.center])[0])
                others = {}
                for b in balls:
                    if b is cue_ball:
                        continue
                    bid = str(b.id)
                    others[bid] = tuple(warp_points(calib.H_cam2table, [b.center])[0])
                    id_colors[bid] = b.color_bgr
                # Aim direction in mm (warp two points along the cue line).
                tip = np.array(cue_ball.center)
                ahead = tip + np.array(cue.direction) * 50.0
                seg = warp_points(calib.H_cam2table, [tuple(tip), tuple(ahead)])
                aim_mm = seg[1] - seg[0]
                aim_mm = aim_mm / (np.linalg.norm(aim_mm) + 1e-9)

                shot = Shot(
                    cue_pos=cue_mm, balls=others,
                    aim_dir=(float(aim_mm[0]), float(aim_mm[1])),
                    speed=controls.strength * cfg.physics.max_speed_mmps,
                    ball_radius=r_mm, table=table, english=controls.english,
                )
                traj = engine.simulate(shot)

            # --- projector overlay ---
            overlay = np.zeros((proj_h, proj_w, 3), np.uint8)
            if traj is not None:
                _draw_paths(overlay, traj, to_proj, id_colors, radius_proj)
            draw_controls(overlay, controls, scale=cfg.controls.ui_scale)
            cv2.putText(overlay, controls.describe(), (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
            sink.show(overlay)

            if args.debug:
                cam = frame.rgb.copy()
                if traj is not None:
                    _draw_paths(cam, traj, to_cam, id_colors, radius_cam)
                draw_controls(cam, controls, origin=(15, cam.shape[0] - 150),
                              scale=0.7)
                cv2.imshow("camera (debug)", cam)

            key = sink.poll_key()
            if args.debug:
                key = (cv2.waitKey(1) & 0xFF) if key == -1 else key
            if key in (27, ord("q")):
                break
            controls.handle_key(key)

    if args.debug:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
