"""Phase 5 entrypoint: projected practice drills with automatic scoring.

Run:
    python -m pool_guide.apps.drills                 # start with a suggested drill
    python -m pool_guide.apps.drills --drill cut_shot
    python -m pool_guide.apps.drills --list          # list drills and your stats
    python -m pool_guide.apps.drills --debug         # + camera-view window

The projector draws where to place the balls, the target pocket(s), and any
cue-ball position zone. Set up the balls, take your shot, and the vision system
scores it: potted? scratched? cue ball left in the zone? Stats and streaks are
tracked to drill_progress.json, and `n` jumps to whatever you should work on next.

Keys:  n next (suggested)   [ ] prev/next drill   SPACE force-ready   r reset   q quit

Requires a full calibration (camera->table) like the shot predictor.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ..calibration.model import warp_points
from ..capture import open_source
from ..config import load_config
from ..display import open_sink
from ..drills import (
    DrillSession,
    Phase,
    ProgressStore,
    get_drill,
    list_drills,
    suggest_next,
)
from ..drills.model import to_mm
from ..drills.session import DetectedBall
from ..recommend import standard_pockets
from ..vision import BallDetector, SimpleTracker, build_table_mask
from .aim_assist import _pick_cue_ball
from .shot_predictor import _require_calibration

GREEN = (0, 235, 0)
WHITE = (255, 255, 255)
RED = (60, 60, 235)
CYAN = (255, 220, 0)


def _print_list(store):
    print("Available drills:")
    for d in list_drills():
        s = store.stats(d.id)
        print(f"  {d.id:14s} [{d.category:8s} d{d.difficulty}]  "
              f"{s.makes}/{s.attempts} ({s.pct:.0f}%)  best streak {s.best_streak}  - {d.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Projected practice drills (Phase 5)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--drill", default=None, help="drill id (default: suggested)")
    ap.add_argument("--list", action="store_true", help="list drills + stats and exit")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    store = ProgressStore(cfg.drills.progress_path)
    if args.list:
        _print_list(store)
        return 0

    calib = _require_calibration(cfg)
    if calib is None:
        return 2

    H_table2cam = np.linalg.inv(calib.H_cam2table)
    H_table2proj = calib.H_cam2proj @ H_table2cam
    proj_w, proj_h = calib.projector_size
    L, W = calib.table_size_mm
    table = (0.0, 0.0, float(L), float(W))
    pockets_mm = standard_pockets(table)

    detector = BallDetector(cfg.vision, calib)
    tracker = SimpleTracker(cfg.vision.tracker_max_dist_px)

    ids = [d.id for d in list_drills()]
    drill_id = args.drill or suggest_next(ids, store)
    session = DrillSession(get_drill(drill_id), table, cfg.drills)

    def to_proj(mm_pts):
        return warp_points(H_table2proj, mm_pts)

    def to_cam(mm_pts):
        return warp_points(H_table2cam, mm_pts)

    def r_at(mm_pt, warp):
        e = warp([mm_pt, (mm_pt[0] + cfg.physics.ball_diameter_mm / 2, mm_pt[1])])
        return max(4, int(np.linalg.norm(e[1] - e[0])))

    def switch(new_id):
        nonlocal session
        session = DrillSession(get_drill(new_id), table, cfg.drills)
        print(f"Drill: {session.drill.name} -- {session.drill.description}")

    print(f"Drill: {session.drill.name} -- {session.drill.description}")

    with open_source(cfg) as src, open_sink(cfg) as sink:
        while True:
            frame = src.read()
            if frame is None:
                if sink.poll_key() in (27, ord("q")):
                    break
                continue

            mask, felt_hue = build_table_mask(frame.rgb, calib, cfg.vision)
            balls = tracker.update(detector.detect(frame.rgb, mask, felt_hue))
            cue_ball = _pick_cue_ball(balls)
            detected = [
                DetectedBall(str(b.id), b is cue_ball,
                             tuple(warp_points(calib.H_cam2table, [b.center])[0]))
                for b in balls
            ]
            session.update(detected)
            if session.new_result:
                store.record(session.drill.id, session.last_result.success, ts=time.time())

            overlay = np.zeros((proj_h, proj_w, 3), np.uint8)
            _render(overlay, session, store, pockets_mm, to_proj,
                    lambda p: r_at(p, to_proj))
            sink.show(overlay)

            if args.debug:
                cam = frame.rgb.copy()
                _render(cam, session, store, pockets_mm, to_cam,
                        lambda p: r_at(p, to_cam), hud_top=True)
                cv2.imshow("camera (debug)", cam)

            key = sink.poll_key()
            if args.debug:
                key = (cv2.waitKey(1) & 0xFF) if key == -1 else key
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                session.force_ready()
            elif key == ord("r"):
                switch(session.drill.id)
            elif key == ord("n"):
                switch(suggest_next(ids, store))
            elif key == ord("]"):
                switch(ids[(ids.index(session.drill.id) + 1) % len(ids)])
            elif key == ord("["):
                switch(ids[(ids.index(session.drill.id) - 1) % len(ids)])

    return 0


def _render(canvas, session, store, pockets_mm, warp, r_at, hud_top=False):
    drill = session.drill
    dim = session.phase is Phase.SHOOTING

    # target pockets
    for idx in drill.target_pockets:
        p = warp([pockets_mm[idx]])[0].astype(int)
        cv2.circle(canvas, tuple(p), r_at(pockets_mm[idx]) + 8, GREEN, 2)

    # cue-ball position zone
    if session.leave_center is not None:
        c = warp([session.leave_center])[0].astype(int)
        # radius in canvas px from the zone radius
        e = warp([session.leave_center,
                  (session.leave_center[0] + session.leave_radius, session.leave_center[1])])
        rad = int(np.linalg.norm(e[1] - e[0]))
        cv2.circle(canvas, tuple(c), max(6, rad), CYAN, 2)

    # placement markers (where to put the balls)
    if session.phase in (Phase.SETUP, Phase.READY):
        if session.cue_spot is not None:
            p = warp([session.cue_spot])[0].astype(int)
            cv2.circle(canvas, tuple(p), r_at(session.cue_spot), WHITE, 2)
            cv2.putText(canvas, "cue", (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        for spot in session.object_spots:
            p = warp([spot])[0].astype(int)
            cv2.circle(canvas, tuple(p), r_at(spot), (0, 165, 255), 2)

    # phase banner
    banner = {
        Phase.SETUP: ("Place the balls on the markers", WHITE),
        Phase.READY: ("Ready -- take your shot", GREEN),
        Phase.SHOOTING: ("...", (150, 150, 150)),
        Phase.RESULT: (None, None),
    }[session.phase]
    if session.phase is Phase.RESULT and session.last_result is not None:
        ok = session.last_result.success
        text = "SUCCESS" if ok else "MISS"
        color = GREEN if ok else RED
        cv2.putText(canvas, text, (canvas.shape[1] // 2 - 90, canvas.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
        cv2.putText(canvas, session.last_result.reason,
                    (canvas.shape[1] // 2 - 140, canvas.shape[0] // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    elif banner[0]:
        cv2.putText(canvas, banner[0], (canvas.shape[1] // 2 - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner[1], 2)

    # HUD: drill + stats
    life = store.stats(drill.id)
    stars = "*" * drill.difficulty
    hud = (f"{drill.name} [{drill.category} {stars}]  "
           f"session {session.makes}/{session.attempts} streak {session.streak}  "
           f"lifetime {life.pct:.0f}%")
    y = 70 if hud_top else canvas.shape[0] - 15
    cv2.putText(canvas, hud, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    if not dim:
        cv2.putText(canvas, drill.description, (15, y - 22 if not hud_top else 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
