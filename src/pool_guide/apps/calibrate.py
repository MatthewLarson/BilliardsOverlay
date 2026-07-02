"""Phase 0 entrypoint: camera <-> projector <-> table calibration.

Run (standalone, from the repo root):
    python -m pool_guide.apps.calibrate

What it does:
  1. Projects a grid of ArUco markers.
  2. Watches the camera and, over several frames, records where each marker id
     lands in camera pixels (keeping the best-detected frame).
  3. Solves the camera->projector homography and reports reprojection error.
  4. (Optional, local only) Lets you click the four table corners to also solve
     the camera->table-millimetres homography for the physics engine.
  5. Writes calibration.json.

Works in distributed mode too: the pattern is streamed to the sensor node's
projector and frames stream back. Table-corner clicking is skipped there
(no local camera window on the brain) -- run with --skip-table or click on the
sensor node in a later pass.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from ..calibration import aruco, save_calibration
from ..capture import open_source
from ..config import load_config
from ..display import open_sink


def _accumulate_detections(cfg, src, sink, pattern, dict_name, warmup, samples):
    """Project the pattern; return the camera-detection dict from the frame with
    the most markers seen. `warmup` frames are discarded so exposure/network settle."""
    best_centers: dict[int, tuple[float, float]] = {}
    best_frame = None
    total = warmup + samples
    for i in range(total):
        sink.show(pattern)
        key = sink.poll_key()
        if key in (27, ord("q")):  # Esc / q aborts
            raise KeyboardInterrupt
        frame = src.read()
        if frame is None:
            continue
        if i < warmup:
            continue
        centers = aruco.detect_marker_centers(frame.rgb, dict_name)
        if len(centers) > len(best_centers):
            best_centers = centers
            best_frame = frame.rgb
        print(f"  frame {i - warmup + 1}/{samples}: {len(centers)} markers", end="\r")
    print()
    return best_centers, best_frame


def _pick_table_corners(image_bgr) -> np.ndarray | None:
    """Let the user click 4 table corners (TL, TR, BR, BL). Returns 4x2 or None."""
    pts: list[tuple[int, int]] = []
    win = "click table corners: TL, TR, BR, BL  (Enter=accept, r=reset, Esc=skip)"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    labels = ["TL", "TR", "BR", "BL"]
    while True:
        canvas = image_bgr.copy()
        for idx, p in enumerate(pts):
            cv2.circle(canvas, p, 6, (0, 255, 0), -1)
            cv2.putText(canvas, labels[idx], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(pts) == 4:
            cv2.polylines(canvas, [np.array(pts)], True, (0, 255, 0), 2)
        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:            # Esc -> skip
            cv2.destroyWindow(win)
            return None
        if k == ord("r"):
            pts.clear()
        if k in (13, 10) and len(pts) == 4:  # Enter
            cv2.destroyWindow(win)
            return np.array(pts, dtype=np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pool Guide camera/projector calibration")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--skip-table", action="store_true",
                    help="don't prompt for table corners (camera->projector only)")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dict_name = cfg.calibration.aruco_dict
    proj_w, proj_h = cfg.display.width, cfg.display.height

    print(f"Mode: {cfg.mode} | capture: {cfg.capture.source} | display: {cfg.display.sink}")
    print(f"Building {cfg.calibration.marker_count}-marker pattern "
          f"for {proj_w}x{proj_h} projector...")
    pattern, proj_centers = aruco.build_projector_pattern(
        proj_w, proj_h, cfg.calibration.marker_count, dict_name)

    with open_source(cfg) as src, open_sink(cfg) as sink:
        print("Projecting pattern and detecting markers (press q/Esc to abort)...")
        try:
            cam_centers, cam_frame = _accumulate_detections(
                cfg, src, sink, pattern, dict_name, args.warmup, args.samples)
        except KeyboardInterrupt:
            print("Aborted.")
            return 1

        matched = sorted(set(cam_centers) & set(proj_centers))
        print(f"Matched {len(matched)} / {cfg.calibration.marker_count} markers.")
        try:
            H_cp, err, n = aruco.solve_cam2proj(cam_centers, proj_centers)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(f"camera->projector homography solved from {n} markers, "
              f"mean reprojection error = {err:.2f} px")

        H_ct = None
        if not args.skip_table and cam_frame is not None and cfg.display.sink != "network":
            print("Click the four table corners to enable real-world coordinates...")
            corners = _pick_table_corners(cam_frame)
            if corners is not None:
                H_ct = aruco.solve_cam2table(
                    corners, cfg.calibration.table_length_mm,
                    cfg.calibration.table_width_mm)
                print("camera->table (mm) homography solved.")
            else:
                print("Skipped table corners.")

        calib = aruco.make_calibration(
            H_cp, err,
            cam_size=(cfg.capture.width, cfg.capture.height),
            proj_size=(proj_w, proj_h),
            table_size_mm=(cfg.calibration.table_length_mm, cfg.calibration.table_width_mm),
            H_cam2table=H_ct,
        )
        save_calibration(calib, cfg.calibration.path)
        print(f"Saved calibration to {cfg.calibration.path}")
        if err > 8.0:
            print("WARNING: reprojection error is high (>8px). Re-run with a rigid "
                  "mount, dimmer room lights, or lower projector brightness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
