"""Built-in practice drills.

Pocket indices match physics/recommend.standard_pockets:
  0 top-left  1 top-right  2 bottom-left  3 bottom-right  4 top-side  5 bottom-side
(normalised corners (0,0)(1,0)(0,1)(1,1); sides (0.5,0)(0.5,1)).
"""
from __future__ import annotations

from .model import BallSpec, Drill, TargetZone

_DRILL_LIST = [
    Drill(
        id="straight_pot",
        name="Straight-in pot",
        category="potting",
        difficulty=1,
        description="Cue, object and top-right pocket are in a line. Pot it clean.",
        balls=[BallSpec("cue", (0.50, 0.50), "cue"),
               BallSpec("object", (0.75, 0.25), "obj")],
        target_pockets=[1],
    ),
    Drill(
        id="stop_shot",
        name="Stop shot",
        category="position",
        difficulty=2,
        description="Pot the ball and stop the cue ball dead at the contact point.",
        balls=[BallSpec("cue", (0.50, 0.50), "cue"),
               BallSpec("object", (0.70, 0.30), "obj")],
        target_pockets=[1],
        cue_leave_zone=TargetZone(center=(0.66, 0.34), radius_mm=170.0),
    ),
    Drill(
        id="cut_shot",
        name="Cut shot",
        category="cut",
        difficulty=3,
        description="Angled cut into the top-right pocket. Find the ghost-ball line.",
        balls=[BallSpec("cue", (0.30, 0.55), "cue"),
               BallSpec("object", (0.62, 0.33), "obj")],
        target_pockets=[1],
    ),
    Drill(
        id="speed_control",
        name="Speed control lane",
        category="speed",
        difficulty=2,
        description="No object ball. Roll the cue ball across and stop it in the zone.",
        balls=[BallSpec("cue", (0.15, 0.50), "cue")],
        target_pockets=[],
        cue_leave_zone=TargetZone(center=(0.80, 0.50), radius_mm=160.0),
        required_pots=0,
    ),
    Drill(
        id="wagon_wheel",
        name="Wagon wheel",
        category="potting",
        difficulty=4,
        description="Object ball in the centre. Pot it into any corner; rotate each rack.",
        balls=[BallSpec("cue", (0.35, 0.50), "cue"),
               BallSpec("object", (0.50, 0.50), "obj")],
        target_pockets=[0, 1, 2, 3],
        required_pots=1,
    ),
]

DRILLS: dict[str, Drill] = {d.id: d for d in _DRILL_LIST}


def list_drills() -> list[Drill]:
    return list(_DRILL_LIST)


def get_drill(drill_id: str) -> Drill:
    if drill_id not in DRILLS:
        raise KeyError(f"Unknown drill {drill_id!r}. Options: {sorted(DRILLS)}")
    return DRILLS[drill_id]
