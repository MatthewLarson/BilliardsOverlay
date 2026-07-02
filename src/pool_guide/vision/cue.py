"""Detect the cue stick and the direction it is aiming.

Approach (2D, from the top-down RGB image):
  1. Canny edges -> HoughLinesP gives many short segments (the two long edges of
     the shaft, plus rails and ball edges).
  2. Merge near-collinear, near-overlapping segments into full-length lines. This
     recovers the shaft as one long line with an accurate orientation.
  3. Pick the line that is long AND passes close to the cue ball -- the cue is
     aimed at the cue ball, so its line nearly intersects the ball's centre.
  4. Orient it: the "tip" is the end nearer the cue ball; the aim direction runs
     from the butt through the tip and on through the ball.

Kinect DEPTH could later refine this (confirm the raised butt vs. the low tip,
and measure cue elevation for jump/masse shots), but the top-down angle -- which
is what the aim line on the cloth needs -- comes from RGB alone. Depth is left
as a Phase 3+ refinement; v1 is RGB-only so it works with any overhead camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Cue:
    tip: tuple[float, float]        # end nearest the cue ball
    butt: tuple[float, float]       # far end
    direction: tuple[float, float]  # unit vector butt -> tip -> ball
    length: float

    @property
    def angle_rad(self) -> float:
        return float(np.arctan2(self.direction[1], self.direction[0]))


def _seg_angle(seg) -> float:
    x1, y1, x2, y2 = seg
    return float(np.arctan2(y2 - y1, x2 - x1))


def _angle_close(a: float, b: float, tol: float) -> bool:
    """True if two undirected line angles are within tol (radians)."""
    d = abs(a - b) % np.pi
    d = min(d, np.pi - d)
    return d <= tol


def _point_line_distance(p, a, b) -> float:
    """Perpendicular distance from point p to the infinite line through a,b."""
    a = np.asarray(a, float); b = np.asarray(b, float); p = np.asarray(p, float)
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-6:
        return float(np.linalg.norm(p - a))
    ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]     # 2D scalar cross product
    return float(abs(cross) / n)


def _merge_segments(segs, angle_tol, dist_tol):
    """Greedily fuse near-collinear, near-overlapping segments into longer ones."""
    merged = []
    used = [False] * len(segs)
    for i, s in enumerate(segs):
        if used[i]:
            continue
        pts = [np.array(s[:2], float), np.array(s[2:], float)]
        ai = _seg_angle(s)
        used[i] = True
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            t = segs[j]
            if not _angle_close(ai, _seg_angle(t), angle_tol):
                continue
            # collinear if endpoints of j sit close to line i
            d = max(_point_line_distance(t[:2], pts_extent(pts)[0], pts_extent(pts)[1]),
                    _point_line_distance(t[2:], pts_extent(pts)[0], pts_extent(pts)[1]))
            if d > dist_tol:
                continue
            pts += [np.array(t[:2], float), np.array(t[2:], float)]
            used[j] = True
        a, b = pts_extent(pts)
        merged.append((a, b))
    return merged


def pts_extent(pts):
    """Return the two most distant points among a cluster (the line's endpoints)."""
    arr = np.array(pts)
    # Project onto the principal direction and take the extremes.
    c = arr.mean(axis=0)
    d = arr - c
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis = vt[0]
    proj = d @ axis
    return arr[int(np.argmin(proj))], arr[int(np.argmax(proj))]


class CueDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, frame_bgr: np.ndarray,
               cue_ball: tuple[float, float] | None) -> Cue | None:
        cfg = self.cfg
        h, w = frame_bgr.shape[:2]
        diag = float(np.hypot(w, h))
        min_len = cfg.cue_min_length_frac * diag

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, cfg.cue_canny_lo, cfg.cue_canny_hi)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=int(min_len * 0.6), maxLineGap=30)
        if lines is None:
            return None
        segs = [tuple(map(float, l[0])) for l in lines]
        merged = _merge_segments(segs, np.deg2rad(cfg.cue_merge_angle_deg),
                                 cfg.cue_merge_dist_px)

        best = None
        best_score = None
        for a, b in merged:
            length = float(np.linalg.norm(b - a))
            if length < min_len:
                continue
            if cue_ball is not None:
                d_ball = _point_line_distance(cue_ball, a, b)
                if d_ball > cfg.cue_max_ball_dist_px:
                    continue
                # Prefer lines that both pass near the ball and are long.
                score = d_ball - 0.05 * length
            else:
                score = -length
            if best_score is None or score < best_score:
                best_score, best = score, (a, b, length)
        if best is None:
            return None

        a, b, length = best
        # Orient: tip = endpoint nearer the cue ball (or the frame interior).
        ref = np.array(cue_ball, float) if cue_ball is not None else np.array([w / 2, h / 2])
        if np.linalg.norm(a - ref) < np.linalg.norm(b - ref):
            tip, butt = a, b
        else:
            tip, butt = b, a
        direction = tip - butt
        n = np.linalg.norm(direction)
        if n < 1e-6:
            return None
        direction = direction / n
        return Cue(tip=tuple(tip), butt=tuple(butt),
                   direction=(float(direction[0]), float(direction[1])),
                   length=length)
