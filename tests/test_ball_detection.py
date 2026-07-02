"""Hardware-free ball-detection tests driven by the synthetic table source.

The synthetic source renders a felt rectangle with black pockets and four
coloured balls (cue + three). We assert the detector finds those balls inside
the table, rejects the pockets, and classifies the cue ball as white.
"""
import numpy as np

from pool_guide.config import load_config
from pool_guide.capture import open_source
from pool_guide.vision import BallDetector, SimpleTracker, build_table_mask


def _frame():
    cfg = load_config()
    cfg.capture.source = "synthetic"
    src = open_source(cfg)
    # advance a few frames so ball positions are well inside the felt
    for _ in range(5):
        f = src.read()
    return f.rgb, cfg


def test_detects_four_balls_and_rejects_pockets():
    rgb, cfg = _frame()
    mask, felt_hue = build_table_mask(rgb, None, cfg.vision)
    assert mask.any(), "table mask should cover the felt"
    assert felt_hue is not None

    detector = BallDetector(cfg.vision, calib=None)
    balls = detector.detect(rgb, mask, felt_hue)

    # Four balls; the six black pockets must NOT be counted.
    assert len(balls) == 4, f"expected 4 balls, got {len(balls)}: {[b.label for b in balls]}"
    # Exactly one cue ball, classified white.
    cues = [b for b in balls if b.label == "cue"]
    assert len(cues) == 1, f"expected 1 cue ball, got {len(cues)}"
    # Radii are sane and consistent.
    radii = [b.radius for b in balls]
    assert all(6 <= r <= 40 for r in radii), radii


def test_tracker_keeps_ids_stable_across_frames():
    cfg = load_config()
    cfg.capture.source = "synthetic"
    src = open_source(cfg)
    detector = BallDetector(cfg.vision, calib=None)
    tracker = SimpleTracker(cfg.vision.tracker_max_dist_px)

    id_sets = []
    for _ in range(6):
        rgb = src.read().rgb
        mask, felt_hue = build_table_mask(rgb, None, cfg.vision)
        balls = tracker.update(detector.detect(rgb, mask, felt_hue))
        id_sets.append({b.id for b in balls})

    # After the first couple of frames the id set should be stable (slow drift).
    assert id_sets[-1] == id_sets[-2]
    assert len(id_sets[-1]) == 4
