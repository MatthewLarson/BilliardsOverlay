"""Typed configuration loading.

Reads config.yaml (falling back to config.example.yaml) into nested dataclasses
so the rest of the codebase gets attribute access and sane defaults instead of
poking at raw dicts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CaptureConfig:
    source: str = "synthetic"          # kinect_v1 | webcam | synthetic | network
    width: int = 640
    height: int = 480
    fps: int = 30
    depth: bool = False
    webcam_index: int = 0
    flip_horizontal: bool = False


@dataclass
class DisplayConfig:
    sink: str = "window"               # projector | window | network
    monitor: int = 1
    width: int = 1280
    height: int = 720


@dataclass
class NetworkConfig:
    role: str = "sensor"               # sensor | brain
    brain_host: str = "127.0.0.1"
    frame_port: int = 5555
    overlay_port: int = 5556
    jpeg_quality: int = 80


@dataclass
class CalibrationConfig:
    path: str = "calibration.json"
    table_length_mm: int = 2540
    table_width_mm: int = 1270
    aruco_dict: str = "DICT_4X4_50"
    marker_count: int = 12


@dataclass
class VisionConfig:
    background_path: str = "background.jpg"   # empty-table reference (optional)
    ball_diameter_mm: float = 57.15           # regulation 2.25in pool ball
    min_ball_radius_px: int = 5               # fallbacks when no table homography
    max_ball_radius_px: int = 40
    radius_tolerance: float = 0.45            # +/- around expected radius when scale is known
    min_circularity: float = 0.6
    felt_hue_tolerance: int = 12              # +/- hue band treated as felt
    min_saturation: int = 40                  # ignore washed-out/gray pixels when finding felt
    min_value: int = 40                       # ignore near-black (pockets/shadows)
    table_erode_px: int = 10                  # shrink the felt mask to stay off rails/pockets
    tracker_max_dist_px: int = 40             # max centroid jump to keep the same ball id
    use_background_subtraction: bool = False  # if true and background exists, prefer it

    # --- Cue detection (Phase 2) ---
    cue_min_length_frac: float = 0.15         # min cue line length as a fraction of frame diagonal
    cue_canny_lo: int = 50
    cue_canny_hi: int = 150
    cue_merge_angle_deg: float = 7.0          # merge line segments within this angle...
    cue_merge_dist_px: float = 25.0           # ...and this perpendicular distance
    cue_max_ball_dist_px: float = 60.0        # cue line must pass this close to the cue ball to count

    # --- Aim preview (Phase 2, pure geometry -- real physics arrives in Phase 3) ---
    aim_max_bounces: int = 3                   # cushion reflections to preview for the cue path
    aim_show_object_dir: bool = True           # draw the struck object ball's predicted direction


@dataclass
class PhysicsConfig:
    engine: str = "simple"             # simple | pooltool
    ball_diameter_mm: float = 57.15    # regulation ball; sets collision radius
    max_speed_mmps: float = 4200.0     # cue-ball speed at strength = 1.0 (~4.2 m/s)
    friction_decel: float = 900.0      # rolling deceleration, mm/s^2 (tune to your cloth)
    restitution_ball: float = 0.96     # ball-ball bounciness
    restitution_cushion: float = 0.80  # energy kept after a cushion hit
    dt: float = 0.001                  # integration timestep, seconds
    max_time: float = 12.0             # give up after this many seconds of sim
    stop_speed: float = 6.0            # below this speed (mm/s) a ball is "stopped"
    pocket_radius_mm: float = 62.0     # capture radius at each pocket
    follow_draw_gain: float = 0.7      # how strongly vertical english pushes/pulls the cue ball
    side_throw_deg: float = 5.0        # max object-ball throw angle at full side english
    sample_every: int = 8             # record every Nth step into the drawn trajectory


@dataclass
class ControlsConfig:
    strength: float = 0.5              # initial strength (0..1)
    strength_step: float = 0.05
    english_step: float = 0.1          # contact-point nudge per keypress
    ui_scale: float = 1.0              # size of the projected meter/contact widgets


@dataclass
class DrillsConfig:
    progress_path: str = "drill_progress.json"
    place_tol_mm: float = 70.0         # how close a ball must be to its spot to count as placed
    motion_thresh_mm: float = 8.0      # per-frame movement that counts as "moving"
    stationary_frames: int = 6         # frames of stillness before "stopped"
    result_hold_frames: int = 45       # how long to show the SUCCESS/MISS verdict
    leave_zone_radius_mm: float = 150.0  # default cue-ball "position" target radius


@dataclass
class Config:
    mode: str = "standalone"           # standalone | distributed
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    controls: ControlsConfig = field(default_factory=ControlsConfig)
    drills: DrillsConfig = field(default_factory=DrillsConfig)


def _build(cls, data: dict[str, Any] | None):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config from `path`, else config.yaml, else config.example.yaml."""
    if path is None:
        root = Path(__file__).resolve().parents[2]
        for candidate in (root / "config.yaml", root / "config.example.yaml"):
            if candidate.exists():
                path = candidate
                break
        else:
            return Config()  # all defaults
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(
        mode=raw.get("mode", "standalone"),
        capture=_build(CaptureConfig, raw.get("capture")),
        display=_build(DisplayConfig, raw.get("display")),
        network=_build(NetworkConfig, raw.get("network")),
        calibration=_build(CalibrationConfig, raw.get("calibration")),
        vision=_build(VisionConfig, raw.get("vision")),
        physics=_build(PhysicsConfig, raw.get("physics")),
        controls=_build(ControlsConfig, raw.get("controls")),
        drills=_build(DrillsConfig, raw.get("drills")),
    )
