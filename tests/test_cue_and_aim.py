"""Phase 2 tests: cue-angle recovery (synthetic) and aim geometry (analytic)."""
import numpy as np

from pool_guide.config import load_config
from pool_guide.capture import open_source
from pool_guide.vision import CueDetector, compute_aim
from pool_guide.vision.aim import inset_convex_quad


def _angle_between(d1, d2):
    a = np.arctan2(d1[1], d1[0])
    b = np.arctan2(d2[1], d2[0])
    diff = abs(a - b) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)


def test_cue_direction_recovered_from_synthetic():
    cfg = load_config()
    cfg.capture.source = "synthetic"
    src = open_source(cfg)
    det = CueDetector(cfg.vision)

    errors = []
    hits = 0
    for _ in range(12):
        frame = src.read()
        truth_dir = src.last_cue_dir
        cue = det.detect(frame.rgb, src.last_cue_ball_px)
        if cue is None:
            continue
        hits += 1
        errors.append(_angle_between(cue.direction, truth_dir))

    assert hits >= 8, f"cue detected in only {hits}/12 frames"
    # Median angular error should be small (within ~6 degrees).
    assert np.median(errors) < np.deg2rad(6), \
        f"median cue angle error {np.rad2deg(np.median(errors)):.1f} deg"


def test_aim_hits_ball_straight_on():
    # Cue ball at origin aiming +x; an object ball dead ahead -> ghost ball should
    # sit exactly 2r short of the object centre, object departs along +x.
    r = 10.0
    quad = np.array([[-500, -300], [500, -300], [500, 300], [-500, 300]], float)
    res = compute_aim((0, 0), (1, 0), [((200, 0), r)], quad, r, max_bounces=2)
    assert res.contact == "ball"
    assert res.ghost_center is not None
    assert abs(res.ghost_center[0] - (200 - 2 * r)) < 1e-3
    assert abs(res.ghost_center[1]) < 1e-3
    assert np.allclose(res.object_dir, (1.0, 0.0), atol=1e-6)


def test_aim_bounces_off_rail():
    # No balls; aim up-right into the top rail -> expect at least one bounce and
    # the reflected segment to head downward again.
    r = 10.0
    quad = np.array([[-500, -300], [500, -300], [500, 300], [-500, 300]], float)
    res = compute_aim((0, 0), (1, -1), [], quad, r, max_bounces=3)
    assert res.bounces >= 1
    assert len(res.rail_points) >= 1
    # First rail contact is near the inset top edge (y = -300 + r).
    assert abs(res.rail_points[0][1] - (-300 + r)) < 1.0


def test_inset_shrinks_quad():
    quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    inset = inset_convex_quad(quad, 10)
    assert np.allclose(inset, [[10, 10], [90, 10], [90, 90], [10, 90]], atol=1e-6)
