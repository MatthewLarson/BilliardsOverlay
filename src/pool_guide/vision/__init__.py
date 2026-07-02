"""Computer vision: table region masking and ball detection (Phase 1)."""
from .aim import AimResult, compute_aim
from .balls import Ball, BallDetector
from .cue import Cue, CueDetector
from .table import (
    build_table_mask,
    px_per_mm,
    table_polygon_camera,
    table_quad_camera,
)
from .tracking import SimpleTracker

__all__ = [
    "Ball",
    "BallDetector",
    "Cue",
    "CueDetector",
    "AimResult",
    "compute_aim",
    "SimpleTracker",
    "build_table_mask",
    "table_polygon_camera",
    "table_quad_camera",
    "px_per_mm",
]
