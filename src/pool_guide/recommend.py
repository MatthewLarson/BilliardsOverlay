"""Best-shot recommendation: search candidate shots, score outcomes, pick one.

For every (target ball, pocket) pair we compute the geometric aim that would
send that ball into that pocket -- the classic "ghost ball" aim: the cue ball
must arrive where its centre sits 2r from the object ball on the line from the
pocket through the object ball. We then simulate a few strengths of that shot
with the physics engine and score the result:

  + potting the target                        (the goal)
  - scratching the cue ball                    (disqualifying)
  + leaving the cue ball away from cushions     (a usable next shot)
  + near-misses count a little                  (so we still rank hopeless racks)

Obstructions need no special handling: if another ball blocks the line, the cue
ball strikes it first in simulation, the target doesn't drop, and the shot
scores low on its own. Thin cuts beyond `max_cut_deg` are pruned before
simulating (they're unreliable and waste search time).

This runs on demand / when the table changes, not every frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import PhysicsEngine, Shot, Trajectory


@dataclass
class ShotCandidate:
    target: str
    pocket: tuple[float, float]
    aim_dir: tuple[float, float]
    speed: float
    english: tuple[float, float]
    ghost: tuple[float, float]
    score: float
    potted_target: bool
    scratch: bool
    cue_leave: tuple[float, float]
    trajectory: Trajectory


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def standard_pockets(table) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = table
    xm = (x0 + x1) / 2
    return [(x0, y0), (x1, y0), (x0, y1), (x1, y1), (xm, y0), (xm, y1)]


def _segment_blocked(a, b, balls: dict, r, exclude) -> bool:
    """True if any ball centre (except `exclude`) lies within 2r of segment a-b,
    i.e. the cue ball would collide with it on the way to the ghost position."""
    a = np.array(a, float); b = np.array(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return False
    thresh = (1.9 * r) ** 2
    for bid, c in balls.items():
        if bid in exclude:
            continue
        c = np.array(c, float)
        t = float(np.clip((c - a) @ ab / L2, 0, 1))
        closest = a + t * ab
        if float((c - closest) @ (c - closest)) < thresh:
            return True
    return False


def aim_to_pot(cue, target, pocket, r):
    """Ghost-ball aim to pot `target` in `pocket`. Returns (aim, ghost, cut_deg)
    or None if the cut is geometrically impossible."""
    cue = np.array(cue, float); target = np.array(target, float); pocket = np.array(pocket, float)
    to_pocket = _unit(pocket - target)
    ghost = target - to_pocket * 2 * r
    aim_vec = ghost - cue
    if np.linalg.norm(aim_vec) < r:            # cue basically on top of the ghost
        return None
    aim = _unit(aim_vec)
    cut_cos = float(np.clip(aim @ to_pocket, -1, 1))
    if cut_cos <= 0:                            # would have to hit through the ball
        return None
    return aim, tuple(ghost), float(np.degrees(np.arccos(cut_cos)))


def score_outcome(traj: Trajectory, target: str, pocket, table, speed, max_speed) -> float:
    potted = target in traj.potted
    scratch = "cue" in traj.potted
    s = 0.0
    if potted:
        s += 100.0
    if scratch:
        s -= 120.0
    x0, y0, x1, y1 = table
    diag = float(np.hypot(x1 - x0, y1 - y0))
    if potted and not scratch:
        cue = np.array(traj.finals["cue"])
        edge = min(cue[0] - x0, x1 - cue[0], cue[1] - y0, y1 - cue[1])
        half = min(x1 - x0, y1 - y0) / 2
        s += 20.0 * float(np.clip(edge / half, 0, 1))          # central leave = flexible
    if not potted:
        tf = np.array(traj.finals.get(target, (0, 0)))
        s += 25.0 * max(0.0, 1.0 - float(np.linalg.norm(tf - np.array(pocket))) / diag)
    s -= 4.0 * (speed / max_speed)                             # prefer softer, controllable shots
    return s


def recommend_shot(cue_pos, balls: dict[str, tuple[float, float]], table,
                   ball_radius, engine: PhysicsEngine, *,
                   targets: list[str] | None = None,
                   pockets=None, speeds=None, english=(0.0, 0.0),
                   max_cut_deg: float = 78.0, max_speed: float | None = None,
                   top_n: int = 1):
    """Return the best ShotCandidate (or a list if top_n>1), or None if no shot
    could even be attempted."""
    targets = list(balls) if targets is None else [t for t in targets if t in balls]
    pockets = standard_pockets(table) if pockets is None else pockets
    speeds = speeds if speeds is not None else [1200.0, 1900.0, 2700.0]
    if max_speed is None:
        max_speed = max(speeds)

    candidates: list[ShotCandidate] = []
    for t in targets:
        for pk in pockets:
            aim_info = aim_to_pot(cue_pos, balls[t], pk, ball_radius)
            if aim_info is None:
                continue
            aim, ghost, cut = aim_info
            if cut > max_cut_deg:
                continue
            # Skip if another ball blocks the cue's path to the ghost position.
            if _segment_blocked(cue_pos, ghost, balls, ball_radius, exclude={t}):
                continue
            for sp in speeds:
                shot = Shot(cue_pos=cue_pos, balls=balls, aim_dir=aim, speed=sp,
                            ball_radius=ball_radius, table=table, english=english)
                traj = engine.simulate(shot)
                sc = score_outcome(traj, t, pk, table, sp, max_speed)
                candidates.append(ShotCandidate(
                    target=t, pocket=tuple(pk), aim_dir=(float(aim[0]), float(aim[1])),
                    speed=sp, english=english, ghost=ghost, score=sc,
                    potted_target=(t in traj.potted), scratch=("cue" in traj.potted),
                    cue_leave=traj.finals.get("cue", cue_pos), trajectory=traj,
                ))
    if not candidates:
        return None if top_n == 1 else []
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0] if top_n == 1 else candidates[:top_n]
