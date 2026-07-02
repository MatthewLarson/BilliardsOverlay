"""Drill session state machine: watch the table, score each attempt.

Phases:
  SETUP     -> project where to place the balls; wait until they're placed and still
  READY     -> layout matches; snapshot ball counts; wait for the cue ball to move
  SHOOTING  -> balls in motion; wait until everything stops
  RESULT    -> score the attempt, hold the verdict, then re-rack (back to SETUP)

Scoring is COUNT-based (objects before vs after, cue present or not), which is
robust to the tracker swapping ids during fast motion -- far safer than trying to
follow a specific ball id through a break of contact.

The session is fed already-detected balls in TABLE-MILLIMETRE coordinates, so it
is fully testable by scripting frames -- no camera required.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .model import Drill, to_mm


class Phase(str, Enum):
    SETUP = "setup"
    READY = "ready"
    SHOOTING = "shooting"
    RESULT = "result"


@dataclass
class DetectedBall:
    id: str
    is_cue: bool
    pos: tuple[float, float]


@dataclass
class AttemptResult:
    success: bool
    potted: int
    scratched: bool
    cue_leave: tuple[float, float] | None
    reason: str


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class DrillSession:
    def __init__(self, drill: Drill, table, cfg):
        self.drill = drill
        self.table = table
        self.cfg = cfg
        self.phase = Phase.SETUP
        self.last_result: AttemptResult | None = None
        self.new_result = False               # True for the one update that produced a result
        self.attempts = 0
        self.makes = 0
        self.streak = 0

        # Precompute target geometry in mm.
        self.cue_spot = to_mm(drill.cue_spec.pos, table) if drill.cue_spec else None
        self.object_spots = [to_mm(s.pos, table) for s in drill.object_specs]
        self.leave_center = (to_mm(drill.cue_leave_zone.center, table)
                             if drill.cue_leave_zone else None)
        self.leave_radius = (drill.cue_leave_zone.radius_mm
                             if drill.cue_leave_zone else cfg.leave_zone_radius_mm)

        self._prev: dict[str, tuple[float, float]] = {}
        self._still = 0
        self._ready_objects = 0
        self._ready_cue = False
        self._hold = 0

    # -- helpers -----------------------------------------------------------
    def _split(self, balls: list[DetectedBall]):
        cue = next((b for b in balls if b.is_cue), None)
        objs = [b for b in balls if not b.is_cue]
        return cue, objs

    def _moving(self, balls: list[DetectedBall]) -> bool:
        cur = {b.id: b.pos for b in balls}
        moved = False
        for bid, p in cur.items():
            if bid in self._prev and _dist(p, self._prev[bid]) > self.cfg.motion_thresh_mm:
                moved = True
                break
        # A change in ball count (a ball potted/appeared) is also motion.
        if len(cur) != len(self._prev):
            moved = True
        self._prev = cur
        return moved

    def _layout_ready(self, cue, objs) -> bool:
        tol = self.cfg.place_tol_mm
        if self.cue_spot is not None:
            if cue is None or _dist(cue.pos, self.cue_spot) > tol:
                return False
        used = set()
        for spot in self.object_spots:
            near = [b for b in objs if b.id not in used and _dist(b.pos, spot) <= tol]
            if not near:
                return False
            near.sort(key=lambda b: _dist(b.pos, spot))
            used.add(near[0].id)
        return True

    # -- main update -------------------------------------------------------
    def update(self, balls: list[DetectedBall]) -> None:
        self.new_result = False
        cue, objs = self._split(balls)
        moving = self._moving(balls)
        if moving:
            self._still = 0
        else:
            self._still += 1
        stopped = self._still >= self.cfg.stationary_frames

        if self.phase is Phase.SETUP:
            if self._layout_ready(cue, objs) and stopped:
                self._ready_objects = len(self.object_spots)
                self._ready_cue = cue is not None
                self.phase = Phase.READY

        elif self.phase is Phase.READY:
            if not self._layout_ready(cue, objs):
                self.phase = Phase.SETUP           # player disturbed the setup
            elif moving:
                self.phase = Phase.SHOOTING

        elif self.phase is Phase.SHOOTING:
            if stopped:
                self._score(cue, objs)
                self.phase = Phase.RESULT
                self._hold = self.cfg.result_hold_frames

        elif self.phase is Phase.RESULT:
            self._hold -= 1
            if self._hold <= 0:
                self.phase = Phase.SETUP

    def _score(self, cue, objs) -> None:
        potted = max(0, self._ready_objects - len(objs))
        scratched = self._ready_cue and cue is None
        need = self.drill.pots_needed()

        leave_ok = True
        if self.leave_center is not None:
            leave_ok = cue is not None and _dist(cue.pos, self.leave_center) <= self.leave_radius

        success = (potted >= need) and (not scratched) and leave_ok

        reasons = []
        if scratched:
            reasons.append("scratched the cue ball")
        if potted < need:
            reasons.append(f"potted {potted}/{need}")
        if self.leave_center is not None and not leave_ok:
            reasons.append("cue ball outside the position zone")
        reason = "clean!" if success else ("; ".join(reasons) or "missed")

        self.last_result = AttemptResult(
            success=success, potted=potted, scratched=scratched,
            cue_leave=(cue.pos if cue else None), reason=reason)
        self.new_result = True
        self.attempts += 1
        if success:
            self.makes += 1
            self.streak += 1
        else:
            self.streak = 0

    def force_ready(self) -> None:
        """Skip the setup wait (e.g. a keypress) -- snapshot whatever is present."""
        self._ready_objects = len(self.object_spots)
        self._ready_cue = True
        self.phase = Phase.READY
