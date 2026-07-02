"""Geometric aim preview: where the cue ball goes, and what it hits first.

This is PURE GEOMETRY -- straight-line travel, perfect mirror bounces off the
cushions, no spin/friction/throw. It's a fast, always-available preview. Phase 3
swaps in `pooltool` for true physics driven by the strength meter and english.

We march a ray from the cue ball centre in the aim direction and, each step,
take whichever comes first: contact with another ball (-> stop, that's the shot)
or a cushion (-> reflect and continue, up to max_bounces). The cushion boundary
is the table quad shrunk inward by one ball radius (the cue ball's centre can't
reach the cloth's edge), so the bounce points are physically where the ball
turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-6


@dataclass
class AimResult:
    path: list[tuple[float, float]]                 # cue-ball centre polyline (camera px)
    contact: str                                    # 'ball' | 'rail' | 'none'
    object_center: tuple[float, float] | None = None   # struck ball centre
    ghost_center: tuple[float, float] | None = None    # cue centre at contact
    object_dir: tuple[float, float] | None = None      # struck ball's departure dir
    bounces: int = 0
    rail_points: list[tuple[float, float]] = field(default_factory=list)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def _ray_circle_t(o, d, c, R):
    """Smallest t>0 where |o + t d - c| = R, else None."""
    oc = o - c
    b = 2 * np.dot(d, oc)
    cc = np.dot(oc, oc) - R * R
    disc = b * b - 4 * cc
    if disc < 0:
        return None
    s = np.sqrt(disc)
    for t in sorted(((-b - s) / 2, (-b + s) / 2)):
        if t > _EPS:
            return t
    return None


def _ray_segment_t(o, d, a, b):
    """t>0 where the ray hits segment a-b, with the segment's inward-facing info."""
    e = b - a
    denom = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(denom) < _EPS:
        return None
    diff = a - o
    t = (diff[0] * (-e[1]) - diff[1] * (-e[0])) / denom
    u = (d[0] * diff[1] - d[1] * diff[0]) / denom
    if t > _EPS and -_EPS <= u <= 1 + _EPS:
        return t
    return None


def inset_convex_quad(quad: np.ndarray, r: float) -> np.ndarray:
    """Shrink a convex quad inward by r (offset every edge toward the centroid)."""
    quad = np.asarray(quad, float)
    g = quad.mean(axis=0)
    n = len(quad)
    offs = []
    for i in range(n):
        a, b = quad[i], quad[(i + 1) % n]
        edge = _unit(b - a)
        normal = np.array([-edge[1], edge[0]])
        if np.dot(normal, g - (a + b) / 2) < 0:
            normal = -normal                     # make it point inward
        offs.append((a + normal * r, edge))
    out = []
    for i in range(n):
        p1, d1 = offs[(i - 1) % n]
        p2, d2 = offs[i]
        # intersect the two offset edge-lines to get the new corner
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(A)) < _EPS:
            out.append(p2)
            continue
        ts = np.linalg.solve(A, p2 - p1)
        out.append(p1 + d1 * ts[0])
    return np.array(out)


def compute_aim(cue_ball, direction, other_balls, table_quad, ball_radius,
                max_bounces=3, show_object_dir=True) -> AimResult:
    """
    cue_ball:     (x, y) camera px, centre of the cue ball
    direction:    (dx, dy) unit aim vector
    other_balls:  list of (center_xy, radius) for every non-cue ball
    table_quad:   4x2 cushion corners (camera px); inset internally by ball_radius
    ball_radius:  cue ball radius (px)
    """
    o = np.array(cue_ball, float)
    d = _unit(np.array(direction, float))
    boundary = inset_convex_quad(np.asarray(table_quad, float), ball_radius)
    edges = [(boundary[i], boundary[(i + 1) % len(boundary)]) for i in range(len(boundary))]
    g = boundary.mean(axis=0)

    path = [tuple(o)]
    rail_points: list[tuple[float, float]] = []
    pos = o
    for bounce in range(max_bounces + 1):
        best_t, kind, payload = None, None, None

        for center, r in other_balls:
            c = np.array(center, float)
            t = _ray_circle_t(pos, d, c, ball_radius + r)
            if t is not None and (best_t is None or t < best_t):
                best_t, kind, payload = t, "ball", c

        for a, b in edges:
            t = _ray_segment_t(pos, d, np.asarray(a, float), np.asarray(b, float))
            if t is not None and (best_t is None or t < best_t):
                best_t, kind, payload = t, "rail", (np.asarray(a, float), np.asarray(b, float))

        if best_t is None:
            return AimResult(path=path, contact="none", bounces=bounce,
                             rail_points=rail_points)

        contact_pt = pos + d * best_t
        path.append(tuple(contact_pt))

        if kind == "ball":
            obj = payload
            obj_dir = _unit(obj - contact_pt) if show_object_dir else None
            return AimResult(
                path=path, contact="ball",
                object_center=tuple(obj), ghost_center=tuple(contact_pt),
                object_dir=None if obj_dir is None else (float(obj_dir[0]), float(obj_dir[1])),
                bounces=bounce, rail_points=rail_points,
            )

        # rail: reflect and continue
        a, b = payload
        edge = _unit(b - a)
        normal = np.array([-edge[1], edge[0]])
        if np.dot(normal, g - contact_pt) < 0:
            normal = -normal
        d = _unit(d - 2 * np.dot(d, normal) * normal)
        rail_points.append(tuple(contact_pt))
        pos = contact_pt + d * 1e-3

    return AimResult(path=path, contact="rail", bounces=max_bounces,
                     rail_points=rail_points)
