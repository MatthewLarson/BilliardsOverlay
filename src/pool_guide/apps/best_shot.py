"""Phase 4 entrypoint: project the recommended "best shot".

Run:
    python -m pool_guide.apps.best_shot            # projects onto the table
    python -m pool_guide.apps.best_shot --debug    # + camera-view window

Detects the balls (no cue stick needed -- this is advice), searches every
target x pocket x strength with the physics engine, and projects the highest
scoring shot: the target ball highlighted, the aim line to its ghost position,
the cue-ball path, and the object ball's path into the pocket. The strength
meter shows the recommended power.

The search runs only when the table changes (or you press SPACE) so the display
stays responsive. Uses a coarse, fast engine for search. Requires a full
calibration (camera->table) like the shot predictor.
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
from ..physics import make_engine
from ..recommend import recommend_shot
from ..ui import ShotControls, draw_controls
from ..vision import BallDetector, SimpleTracker, build_table_mask
from .aim_assist import _pick_cue_ball
from .shot_predictor import _draw_paths, _require_calibration

GREEN = (0, 235, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 215, 255)


def _search_engine(cfg):
    """A coarse clone of the physics config -- big timestep, short horizon."""
    from dataclasses import replace
    coarse = replace(cfg.physics, dt=max(cfg.physics.dt, 0.0035),
                     max_time=min(cfg.physics.max_time, 8.0),
                     sample_every=max(cfg.physics.sample_every, 20))
    return make_engine(coarse)


def _positions_changed(prev, balls, thresh) -> bool:
    if prev is None or len(prev) != len(balls):
        return True
    cur = {str(b.id): b.center for b in balls}
    if set(cur) != set(prev):
        return True
    return any(np.hypot(cur[k][0] - prev[k][0], cur[k][1] - prev[k][1]) > thresh for k in cur)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Project the best-shot recommendation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--include-8", action="store_true",
                    help="allow recommending the 8-ball even if other balls remain")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = _require_calibration(cfg)
    if calib is None:
        return 2

    H_table2cam = np.linalg.inv(calib.H_cam2table)
    H_table2proj = calib.H_cam2proj @ H_table2cam
    proj_w, proj_h = calib.projector_size
    L, W = calib.table_size_mm
    table = (0.0, 0.0, float(L), float(W))
    r_mm = cfg.physics.ball_diameter_mm / 2

    detector = BallDetector(cfg.vision, calib)
    tracker = SimpleTracker(cfg.vision.tracker_max_dist_px)
    engine = _search_engine(cfg)
    controls = ShotControls.from_config(cfg.controls)

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

    best = None
    prev_positions = None
    id_colors: dict[str, tuple] = {}

    with open_source(cfg) as src, open_sink(cfg) as sink:
        print("Best-shot assistant. SPACE recompute, q/Esc quit.")
        while True:
            frame = src.read()
            if frame is None:
                if sink.poll_key() in (27, ord("q")):
                    break
                continue

            mask, felt_hue = build_table_mask(frame.rgb, calib, cfg.vision)
            balls = tracker.update(detector.detect(frame.rgb, mask, felt_hue))
            cue_ball = _pick_cue_ball(balls)

            key = sink.poll_key()
            if args.debug:
                key = (cv2.waitKey(1) & 0xFF) if key == -1 else key
            force = key == ord(" ")

            if cue_ball is not None and (force or _positions_changed(prev_positions, balls, 12.0)):
                cue_mm = tuple(warp_points(calib.H_cam2table, [cue_ball.center])[0])
                targets_mm = {}
                id_colors = {}
                for b in balls:
                    if b is cue_ball:
                        continue
                    if b.label == "8" and not args.include_8 and len(balls) > 2:
                        continue
                    bid = str(b.id)
                    targets_mm[bid] = tuple(warp_points(calib.H_cam2table, [b.center])[0])
                    id_colors[bid] = b.color_bgr
                best = recommend_shot(
                    cue_mm, targets_mm, table, r_mm, engine,
                    speeds=[cfg.physics.max_speed_mmps * f for f in (0.32, 0.5, 0.72)],
                    max_speed=cfg.physics.max_speed_mmps,
                ) if targets_mm else None
                if best is not None:
                    controls.strength = best.speed / cfg.physics.max_speed_mmps
                prev_positions = {str(b.id): b.center for b in balls}

            # --- render ---
            overlay = np.zeros((proj_h, proj_w, 3), np.uint8)
            _render(overlay, best, to_proj, id_colors, radius_proj, controls)
            sink.show(overlay)

            if args.debug:
                cam = frame.rgb.copy()
                _render(cam, best, to_cam, id_colors, radius_cam, controls, hud_top=True)
                cv2.imshow("camera (debug)", cam)

            if key in (27, ord("q")):
                break
    if args.debug:
        cv2.destroyAllWindows()
    return 0


def _render(canvas, best, warp, id_colors, radius_fn, controls, hud_top=False):
    if best is not None:
        _draw_paths(canvas, best.trajectory, warp, id_colors, radius_fn)
        # highlight the target ball and the ghost-ball aim
        tgt = best.trajectory.paths.get(best.target)
        if tgt:
            tp = warp([tgt[0]])[0].astype(int)
            cv2.circle(canvas, tuple(tp), radius_fn(tgt[0]) + 6, GREEN, 2)
        g = warp([best.ghost])[0].astype(int)
        cv2.drawMarker(canvas, tuple(g), GREEN, cv2.MARKER_TILTED_CROSS, 18, 2)
        pk = warp([best.pocket])[0].astype(int)
        cv2.circle(canvas, tuple(pk), radius_fn(best.pocket) + 8, GREEN, 2)
        verdict = "SCRATCH RISK" if best.scratch else ("POT" if best.potted_target else "best available")
        msg = f"Best shot: ball {best.target}  [{verdict}]  power {controls.strength * 100:.0f}%"
    else:
        msg = "No shot found -- clear the cue ball's view or press SPACE"
    y = 30 if hud_top else canvas.shape[0] - 20
    cv2.putText(canvas, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    draw_controls(canvas, controls, scale=0.9)


if __name__ == "__main__":
    raise SystemExit(main())
