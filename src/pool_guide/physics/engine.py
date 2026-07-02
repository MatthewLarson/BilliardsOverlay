"""Shared physics contract.

Coordinates are a single consistent 2D metric space -- millimetres on the table
when calibration provides camera->table, or pixels otherwise. The engine doesn't
care which, as long as radius, table, speed, and positions all use the same unit.

A Shot fully specifies an attempt:
  cue_pos      cue-ball centre
  balls        the other balls (id -> centre)
  aim_dir      unit vector the cue ball launches along
  speed        launch speed (units/second) -- comes from the strength meter
  english      (a, b): horizontal and vertical contact offset on the cue ball,
               each in [-1, 1]. a = side spin (left/right), b = follow(+)/draw(-).
  ball_radius  collision radius
  table        (xmin, ymin, xmax, ymax) cushion rectangle
  pockets      optional list of pocket centres (defaults to the 6 standard spots)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BallState:
    id: str
    pos: tuple[float, float]
    potted: bool = False


@dataclass
class Shot:
    cue_pos: tuple[float, float]
    balls: dict[str, tuple[float, float]]
    aim_dir: tuple[float, float]
    speed: float
    ball_radius: float
    table: tuple[float, float, float, float]
    english: tuple[float, float] = (0.0, 0.0)
    pockets: list[tuple[float, float]] | None = None

    def pocket_positions(self) -> list[tuple[float, float]]:
        if self.pockets is not None:
            return self.pockets
        x0, y0, x1, y1 = self.table
        xm = (x0 + x1) / 2
        return [(x0, y0), (x1, y0), (x0, y1), (x1, y1), (xm, y0), (xm, y1)]


@dataclass
class Trajectory:
    """Result of a simulated Shot.

    paths:          id -> list of (x, y) sampled centres (id 'cue' is the cue ball)
    potted:         ids that fell in a pocket
    finals:         id -> final resting centre
    cue_first_hit:  id of the first ball the cue ball struck, or None
    ghost:          cue-ball centre at that first contact (the "ghost ball"), or None
    """
    paths: dict[str, list[tuple[float, float]]]
    potted: set[str] = field(default_factory=set)
    finals: dict[str, tuple[float, float]] = field(default_factory=dict)
    cue_first_hit: str | None = None
    ghost: tuple[float, float] | None = None


class PhysicsEngine:
    def simulate(self, shot: Shot) -> Trajectory:
        raise NotImplementedError
