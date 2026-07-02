"""ArUco-based projector<->camera calibration.

Strategy (see interactive-pool / GOSAI for the same idea):
  1. Draw a grid of ArUco markers into a PROJECTOR-space image at known pixel
     centers, keyed by marker id.
  2. Project it; capture the camera's view; detect the markers -> camera-space
     centers, keyed by the same ids.
  3. Every id we see in both spaces is a correspondence (camera px, projector px).
     Solve a homography camera->projector from them (RANSAC).

Also provides a table homography (camera px -> real mm) from the four table
corners, so downstream physics gets real-world coordinates.

Handles both the modern (OpenCV >= 4.7 `ArucoDetector`) and legacy
(`detectMarkers`) cv2.aruco APIs.
"""
from __future__ import annotations

import cv2
import numpy as np

from .model import Calibration


def get_aruco_dict(name: str):
    const = getattr(cv2.aruco, name, None)
    if const is None:
        raise ValueError(f"Unknown ArUco dict {name!r} (e.g. 'DICT_4X4_50')")
    # getPredefinedDictionary is stable across versions.
    return cv2.aruco.getPredefinedDictionary(const)


def _grid_shape(n: int) -> tuple[int, int]:
    """Rows x cols that comfortably holds n markers, wider than tall."""
    cols = int(np.ceil(np.sqrt(n * 16 / 9)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def build_projector_pattern(proj_w: int, proj_h: int, n_markers: int,
                            dict_name: str):
    """Return (bgr_image, centers) where centers maps id -> (px, py) in projector px."""
    adict = get_aruco_dict(dict_name)
    rows, cols = _grid_shape(n_markers)
    img = np.zeros((proj_h, proj_w, 3), np.uint8)

    # Cell geometry with margins so markers don't touch screen edges.
    cell_w = proj_w / (cols + 1)
    cell_h = proj_h / (rows + 1)
    # Bigger markers survive Chromecast compression + a low-res overhead camera.
    marker_px = int(min(cell_w, cell_h) * 0.8)
    centers: dict[int, tuple[float, float]] = {}

    mid = 0
    for r in range(rows):
        for c in range(cols):
            if mid >= n_markers:
                break
            cx = int((c + 1) * cell_w)
            cy = int((r + 1) * cell_h)
            marker = _draw_marker(adict, mid, marker_px)
            x0, y0 = cx - marker_px // 2, cy - marker_px // 2
            img[y0:y0 + marker_px, x0:x0 + marker_px] = marker
            centers[mid] = (cx, cy)
            mid += 1
    return img, centers


def _draw_marker(adict, marker_id: int, size: int) -> np.ndarray:
    """Return a `size`x`size` BGR tile: the marker on a WHITE quiet zone.

    ArUco detection needs a light border around the marker's black frame, so we
    render the marker at ~70% and pad the rest with white. Without this the
    black marker border merges into a black projector background and nothing is
    detected.
    """
    inner = max(8, int(size * 0.7))
    if hasattr(cv2.aruco, "generateImageMarker"):
        m = cv2.aruco.generateImageMarker(adict, marker_id, inner)
    else:  # pragma: no cover - legacy OpenCV
        m = cv2.aruco.drawMarker(adict, marker_id, inner)
    tile = np.full((size, size), 255, np.uint8)  # white quiet zone
    off = (size - inner) // 2
    tile[off:off + inner, off:off + inner] = m
    return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)


def detect_marker_centers(image_bgr: np.ndarray, dict_name: str):
    """Detect markers in a camera frame. Return {id: (cx, cy)} camera px."""
    adict = get_aruco_dict(dict_name)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(adict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # pragma: no cover - legacy OpenCV
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)

    centers: dict[int, tuple[float, float]] = {}
    if ids is None:
        return centers
    for mid, quad in zip(ids.flatten(), corners):
        c = quad.reshape(4, 2).mean(axis=0)
        centers[int(mid)] = (float(c[0]), float(c[1]))
    return centers


def solve_cam2proj(cam_centers: dict[int, tuple[float, float]],
                   proj_centers: dict[int, tuple[float, float]]):
    """Solve homography camera px -> projector px. Return (H, mean_error_px, n)."""
    ids = sorted(set(cam_centers) & set(proj_centers))
    if len(ids) < 4:
        raise RuntimeError(
            f"Need >=4 matched markers to solve homography, got {len(ids)}. "
            "Improve lighting/contrast or reduce projector brightness glare."
        )
    src = np.array([cam_centers[i] for i in ids], dtype=np.float64)
    dst = np.array([proj_centers[i] for i in ids], dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("findHomography failed (degenerate marker layout?)")

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
    return H, err, len(ids)


def solve_cam2table(table_corners_cam: np.ndarray,
                    table_len_mm: int, table_wid_mm: int) -> np.ndarray:
    """Homography from 4 clicked table corners (camera px) to table mm.

    Corners order: top-left, top-right, bottom-right, bottom-left.
    """
    src = np.asarray(table_corners_cam, dtype=np.float64).reshape(4, 2)
    dst = np.array([
        [0, 0],
        [table_len_mm, 0],
        [table_len_mm, table_wid_mm],
        [0, table_wid_mm],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Table homography failed (are the 4 corners valid?)")
    return H


def make_calibration(H_cam2proj, err, cam_size, proj_size, table_size_mm,
                     H_cam2table=None) -> Calibration:
    return Calibration(
        H_cam2proj=H_cam2proj,
        H_cam2table=H_cam2table,
        camera_size=cam_size,
        projector_size=proj_size,
        table_size_mm=table_size_mm,
        reproj_error_px=err,
    )
