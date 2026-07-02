"""Optional high-fidelity backend using pooltool (https://github.com/ekiefl/pooltool).

pooltool is a research-grade, event-based billiards simulator. It is NOT
installed by default because it pins `panda3d`, which currently has no wheel for
Python 3.14 (and is heavy on the Pi). To use this backend:

    * run the brain on Python 3.10-3.12
    * pip install "pooltool-billiards"
    * set physics.engine: pooltool

This adapter maps our unit-agnostic Shot (which must be in METRES here -- see
below) onto pooltool's API and reads the simulated trajectories back out.

NOTE: pooltool works in metres with a specific table model. When this backend is
selected, run the app with table->mm calibration so we can convert mm -> m. The
mapping below targets pooltool's documented API (System / Cue / simulate); if a
future pooltool release changes signatures, adjust here -- the rest of Pool Guide
is insulated from it by the PhysicsEngine interface.
"""
from __future__ import annotations

import numpy as np

from .engine import PhysicsEngine, Shot, Trajectory


class PooltoolEngine(PhysicsEngine):
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            import pooltool as pt  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "physics.engine is 'pooltool' but pooltool is not installed. "
                "Use Python <=3.12 and `pip install pooltool-billiards`, or set "
                "physics.engine: simple. See physics/pooltool_engine.py for notes."
            ) from e
        self._pt = pt

    def simulate(self, shot: Shot) -> Trajectory:  # pragma: no cover - needs pooltool
        pt = self._pt
        mm_to_m = 1e-3

        # Build balls (mm -> m). pooltool expects a dict of pt.Ball by id.
        def m(p):
            return (p[0] * mm_to_m, p[1] * mm_to_m)

        balls = {"cue": pt.Ball.create("cue", xy=m(shot.cue_pos))}
        for bid, pos in shot.balls.items():
            balls[bid] = pt.Ball.create(bid, xy=m(pos))

        table = pt.Table.default()
        cue = pt.Cue(cue_ball_id="cue")
        system = pt.System(table=table, balls=balls, cue=cue)

        # Aim azimuth in degrees; english a/b in [-1,1]; V0 from strength (m/s).
        phi = float(np.rad2deg(np.arctan2(shot.aim_dir[1], shot.aim_dir[0]))) % 360
        a, b = shot.english
        system.cue.set_state(V0=shot.speed * mm_to_m, phi=phi, a=float(a), b=float(b))

        pt.simulate(system, inplace=True)

        # Read continuous histories back out, converting m -> mm.
        paths: dict[str, list[tuple[float, float]]] = {}
        finals: dict[str, tuple[float, float]] = {}
        for bid, ball in system.balls.items():
            hist = ball.history_cts if ball.history_cts else ball.history
            xy = np.array([state.rvw[0][:2] for state in hist]) / mm_to_m
            paths[bid] = [tuple(p) for p in xy]
            finals[bid] = tuple(xy[-1]) if len(xy) else shot.balls.get(bid, shot.cue_pos)

        return Trajectory(paths=paths, potted=set(), finals=finals)
