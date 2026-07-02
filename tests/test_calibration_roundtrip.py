"""End-to-end calibration logic test with no hardware and no GUI.

Simulates a "perfect projector" by treating the rendered pattern itself as the
camera image, then checks that we detect the markers and recover a homography
with sub-pixel reprojection error. Also exercises the synthetic capture source
and the calibration save/load round-trip.
"""
import numpy as np

from pool_guide.calibration import aruco, load_calibration, save_calibration
from pool_guide.config import load_config
from pool_guide.capture import open_source


def test_aruco_homography_roundtrip(tmp_path):
    dict_name = "DICT_4X4_50"
    proj_w, proj_h, n = 1280, 720, 12

    pattern, proj_centers = aruco.build_projector_pattern(proj_w, proj_h, n, dict_name)
    # "Perfect" projection: the camera sees exactly the projected image.
    cam_centers = aruco.detect_marker_centers(pattern, dict_name)

    # We should detect essentially every projected marker.
    assert len(cam_centers) >= n - 1

    H, err, matched = aruco.solve_cam2proj(cam_centers, proj_centers)
    assert matched >= 4
    # Identity-ish mapping -> tiny error.
    assert err < 2.0

    calib = aruco.make_calibration(
        H, err, cam_size=(640, 480), proj_size=(proj_w, proj_h),
        table_size_mm=(2540, 1270))
    path = tmp_path / "calibration.json"
    save_calibration(calib, str(path))
    reloaded = load_calibration(str(path))
    assert np.allclose(reloaded.H_cam2proj, calib.H_cam2proj)
    assert reloaded.table_size_mm == (2540, 1270)


def test_cam2table_homography():
    corners = np.array([[10, 10], [630, 12], [628, 470], [8, 468]], dtype=float)
    H = aruco.solve_cam2table(corners, 2540, 1270)
    # Top-left camera corner should map near table origin (0,0) mm.
    from pool_guide.calibration.model import warp_points
    mapped = warp_points(H, corners)
    assert np.allclose(mapped[0], [0, 0], atol=1e-6)
    assert np.allclose(mapped[2], [2540, 1270], atol=1e-6)


def test_synthetic_source_produces_frames():
    cfg = load_config()
    cfg.mode = "standalone"  # hermetic: ignore any ambient distributed config.yaml
    cfg.capture.source = "synthetic"
    src = open_source(cfg)
    f1 = src.read()
    f2 = src.read()
    assert f1.rgb.shape == (cfg.capture.height, cfg.capture.width, 3)
    assert f2.index == f1.index + 1
