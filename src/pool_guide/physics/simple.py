"""A compact, dependency-free billiards simulator.

Models the parts that matter for shot prediction:
  * rolling deceleration (balls slow and stop),
  * ball-ball collisions (equal mass, restitution) -> correct object-ball line,
  * cushion bounces (axis-aligned rectangle, restitution),
  * english: vertical (follow/draw) pushes/pulls the cue ball after it strikes an
    object ball; horizontal (side) throws the object ball's line a few degrees,
  * pocket capture.

It is intentionally NOT a spin-accurate simulator (no throw-from-cut, no
swerve/masse, no cling). It's fast, deterministic, and good enough to make the
strength meter and contact point meaningfully change the predicted outcome.
Swap in the pooltool backend for research-grade fidelity.
"""
from __future__ import annotations

import numpy as np

from .engine import PhysicsEngine, Shot, Trajectory


class SimplePhysicsEngine(PhysicsEngine):
    def __init__(self, cfg):
        self.cfg = cfg

    def simulate(self, shot: Shot) -> Trajectory:
        cfg = self.cfg
        r = shot.ball_radius
        x0, y0, x1, y1 = shot.table
        lo = np.array([x0 + r, y0 + r])
        hi = np.array([x1 - r, y1 - r])

        ids = ["cue"] + list(shot.balls.keys())
        P = np.array([shot.cue_pos] + list(shot.balls.values()), dtype=float)
        V = np.zeros_like(P)
        aim = np.array(shot.aim_dir, float)
        aim = aim / (np.linalg.norm(aim) + 1e-12)
        V[0] = aim * shot.speed
        n = len(ids)

        pockets = np.array(shot.pocket_positions(), float)
        pocket_r = cfg.pocket_radius_mm
        a_eng, b_eng = shot.english

        active = np.ones(n, bool)
        paths: dict[str, list[tuple[float, float]]] = {i: [tuple(p)] for i, p in zip(ids, P)}
        potted: set[str] = set()
        cue_first_hit: str | None = None
        ghost: tuple[float, float] | None = None

        dt = cfg.dt
        decel = cfg.friction_decel
        e_b = cfg.restitution_ball
        e_c = cfg.restitution_cushion
        max_steps = int(cfg.max_time / dt)
        throw_rad = np.deg2rad(cfg.side_throw_deg)

        for step in range(max_steps):
            # --- rolling friction ---
            speed = np.linalg.norm(V, axis=1)
            moving = active & (speed > 1e-9)
            if moving.any():
                new_speed = np.maximum(speed[moving] - decel * dt, 0.0)
                V[moving] *= (new_speed / speed[moving])[:, None]

            # --- integrate ---
            P[active] += V[active] * dt

            # --- cushions (axis-aligned) ---
            for i in np.where(active)[0]:
                for ax in (0, 1):
                    if P[i, ax] < lo[ax]:
                        P[i, ax] = lo[ax]
                        V[i, ax] = -V[i, ax] * e_c
                    elif P[i, ax] > hi[ax]:
                        P[i, ax] = hi[ax]
                        V[i, ax] = -V[i, ax] * e_c

            # --- ball-ball collisions ---
            act_idx = np.where(active)[0]
            for ii in range(len(act_idx)):
                for jj in range(ii + 1, len(act_idx)):
                    i, j = act_idx[ii], act_idx[jj]
                    d = P[j] - P[i]
                    dist = float(np.linalg.norm(d))
                    if dist >= 2 * r or dist < 1e-9:
                        continue
                    nrm = d / dist
                    v1n = float(V[i] @ nrm)
                    v2n = float(V[j] @ nrm)
                    if v1n - v2n <= 0:          # separating -> ignore
                        continue
                    cue_vel_pre = V[i].copy() if i == 0 else None
                    s = v1n + v2n
                    dvn = v1n - v2n
                    V[i] += ((s - e_b * dvn) / 2 - v1n) * nrm
                    V[j] += ((s + e_b * dvn) / 2 - v2n) * nrm
                    overlap = 2 * r - dist
                    P[i] -= nrm * overlap / 2
                    P[j] += nrm * overlap / 2

                    if cue_first_hit is None and i == 0:
                        cue_first_hit = ids[j]
                        ghost = tuple(P[0])
                        self._apply_english(V, i=0, j=j, pre_vel=cue_vel_pre,
                                            a=a_eng, b=b_eng,
                                            gain=cfg.follow_draw_gain, throw=throw_rad)

            # --- pockets ---
            for i in np.where(active)[0]:
                if np.min(np.linalg.norm(pockets - P[i], axis=1)) <= pocket_r:
                    active[i] = False
                    V[i] = 0
                    potted.add(ids[i])
                    paths[ids[i]].append(tuple(P[i]))

            # --- record + stop test ---
            if step % cfg.sample_every == 0:
                for i in np.where(active)[0]:
                    paths[ids[i]].append(tuple(P[i]))
            if not active.any():
                break
            if float(np.max(np.linalg.norm(V[active], axis=1))) < cfg.stop_speed:
                break

        finals = {}
        for i, bid in enumerate(ids):
            paths[bid].append(tuple(P[i]))
            finals[bid] = tuple(P[i])
        return Trajectory(paths=paths, potted=potted, finals=finals,
                          cue_first_hit=cue_first_hit, ghost=ghost)

    @staticmethod
    def _apply_english(V, i, j, pre_vel, a, b, gain, throw):
        """Vertical english pushes/pulls the cue ball; side english throws the object."""
        pre_speed = float(np.linalg.norm(pre_vel))
        if pre_speed < 1e-9:
            return
        d = pre_vel / pre_speed
        # follow (b>0): cue continues along its pre-hit direction; draw (b<0): reverses.
        # Scaled by the incoming speed so it works even on a dead-straight stun.
        V[i] += b * gain * pre_speed * d
        # side throw on the object ball: rotate its outgoing velocity by a*throw.
        if abs(a) > 1e-6 and np.linalg.norm(V[j]) > 1e-6:
            ang = a * throw
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            V[j] = R @ V[j]
