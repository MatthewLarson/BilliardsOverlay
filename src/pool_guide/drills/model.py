"""Drill data model.

Positions are stored NORMALISED (0..1 along table length x width) so a drill
scales to any table; convert to millimetres with `to_mm` given the table size.
`x` runs along the long rail (0 = left/head-string side), `y` across the short
rail (0 = top rail as seen from above).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BallSpec:
    role: str                      # "cue" or "object"
    pos: tuple[float, float]       # normalised (fx, fy)
    label: str = ""                # optional human label ("cue", "1", ...)


@dataclass
class TargetZone:
    """A region the cue ball should come to rest in (position play)."""
    center: tuple[float, float]    # normalised
    radius_mm: float = 150.0


@dataclass
class Drill:
    id: str
    name: str
    category: str                  # potting | position | cut | speed | break
    difficulty: int                # 1 (easy) .. 5 (hard)
    description: str
    balls: list[BallSpec]
    target_pockets: list[int] = field(default_factory=list)  # indices into standard_pockets
    cue_leave_zone: TargetZone | None = None
    required_pots: int | None = None    # defaults to the number of object balls

    @property
    def object_specs(self) -> list[BallSpec]:
        return [b for b in self.balls if b.role == "object"]

    @property
    def cue_spec(self) -> BallSpec | None:
        for b in self.balls:
            if b.role == "cue":
                return b
        return None

    def pots_needed(self) -> int:
        return self.required_pots if self.required_pots is not None else len(self.object_specs)


def to_mm(frac: tuple[float, float], table) -> tuple[float, float]:
    x0, y0, x1, y1 = table
    return (x0 + frac[0] * (x1 - x0), y0 + frac[1] * (y1 - y0))
