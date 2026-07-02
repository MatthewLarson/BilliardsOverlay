"""Phase 0 verification: prove the camera->projector mapping is accurate.

Run:
    python -m pool_guide.apps.verify_calibration

Live loop: re-detects ArUco markers in the camera each frame, maps each detected
center through H_cam2proj, and draws a green reticle there in projector space.
If calibration is good, every reticle sits dead-center on its marker. Physically
nudge a printed ArUco marker around the table and its reticle should track it.

Also overlays the mapped table outline (if a table homography was saved) so you
can eyeball whether projected lines land on the real rails.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..calibration import load_calibration
from ..calibration.aruco import detect_marker_centers
from ..calibration.model import warp_points
from ..capture import open_source
from ..config import load_config
from ..display import open_sink


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify Pool Guide calibration")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = load_calibration(cfg.calibration.path)
    proj_w, proj_h = calib.projector_size
    dict_name = cfg.calibration.aruco_dict
    print(f"Loaded calibration (reproj error {calib.reproj_error_px:.2f}px). "
          "Press q/Esc to quit.")

    # Precompute the table outline in projector space if we have a table homography.
    table_outline_proj = None
    if calib.H_cam2table is not None:
        H_table2proj = calib.H_cam2proj @ np.linalg.inv(calib.H_cam2table)
        L, W = calib.table_size_mm
        corners_mm = np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=np.float64)
        table_outline_proj = warp_points(H_table2proj, corners_mm).astype(np.int32)

    with open_source(cfg) as src, open_sink(cfg) as sink:
        while True:
            frame = src.read()
            overlay = np.zeros((proj_h, proj_w, 3), np.uint8)

            if frame is not None:
                centers = detect_marker_centers(frame.rgb, dict_name)
                if centers:
                    ids = sorted(centers)
                    cam_pts = np.array([centers[i] for i in ids], dtype=np.float64)
                    proj_pts = warp_points(calib.H_cam2proj, cam_pts)
                    for i, (px, py) in zip(ids, proj_pts):
                        p = (int(px), int(py))
                        cv2.circle(overlay, p, 18, (0, 255, 0), 2)
                        cv2.drawMarker(overlay, p, (0, 255, 0), cv2.MARKER_CROSS, 24, 1)
                        cv2.putText(overlay, str(i), (p[0] + 12, p[1] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(overlay, f"markers: {len(centers)}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if table_outline_proj is not None:
                cv2.polylines(overlay, [table_outline_proj], True, (0, 180, 255), 2)

            sink.show(overlay)
            if sink.poll_key() in (27, ord("q")):
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
