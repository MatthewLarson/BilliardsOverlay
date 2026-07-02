"""Give balls stable ids across frames via greedy nearest-neighbour matching.

Not a Kalman filter -- just enough continuity that a projected dot/label stays
attached to the same ball while it's stationary or rolling slowly, which is all
Phase 1 needs. Detections that can't be matched within `max_dist` px get a new
id; ids that vanish for a few frames are forgotten.
"""
from __future__ import annotations

from .balls import Ball


class SimpleTracker:
    def __init__(self, max_dist: float, forget_after: int = 5):
        self.max_dist = max_dist
        self.forget_after = forget_after
        self._next_id = 0
        # id -> (cx, cy, frames_since_seen)
        self._tracks: dict[int, tuple[float, float, int]] = {}

    def update(self, balls: list[Ball]) -> list[Ball]:
        prev = {i: (x, y) for i, (x, y, _) in self._tracks.items()}
        used: set[int] = set()

        # Greedy: match each detection to the closest unused prior track.
        for ball in balls:
            best_id, best_d = None, self.max_dist
            for tid, (px, py) in prev.items():
                if tid in used:
                    continue
                d = ((ball.cx - px) ** 2 + (ball.cy - py) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            ball.id = best_id
            used.add(best_id)
            self._tracks[best_id] = (ball.cx, ball.cy, 0)

        # Age out tracks we didn't see this frame.
        for tid in list(self._tracks):
            if tid not in used:
                x, y, age = self._tracks[tid]
                if age + 1 > self.forget_after:
                    del self._tracks[tid]
                else:
                    self._tracks[tid] = (x, y, age + 1)
        return balls
