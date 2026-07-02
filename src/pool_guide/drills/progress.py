"""Persistent drill progress + a simple coaching suggestion.

Stores a flat list of attempts to JSON. Aggregates per-drill make rate and
streaks, and suggests what to practise next: unattempted drills first, then the
lowest make-rate drill with enough attempts (your weak spot). Timestamps are
passed in by the caller so this stays deterministic and testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DrillStats:
    attempts: int = 0
    makes: int = 0
    streak: int = 0
    best_streak: int = 0

    @property
    def pct(self) -> float:
        return 100.0 * self.makes / self.attempts if self.attempts else 0.0


class ProgressStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self._records: list[dict] = []
        if self.path.exists():
            try:
                self._records = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self._records = []

    def record(self, drill_id: str, success: bool, ts: float = 0.0) -> None:
        self._records.append({"drill": drill_id, "success": bool(success), "ts": ts})
        self._save()

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        except OSError:
            pass

    def stats(self, drill_id: str) -> DrillStats:
        s = DrillStats()
        streak = 0
        for r in self._records:
            if r["drill"] != drill_id:
                continue
            s.attempts += 1
            if r["success"]:
                s.makes += 1
                streak += 1
                s.best_streak = max(s.best_streak, streak)
            else:
                streak = 0
        s.streak = streak
        return s

    def all_stats(self, drill_ids) -> dict[str, DrillStats]:
        return {d: self.stats(d) for d in drill_ids}


def suggest_next(drill_ids, store: ProgressStore, min_attempts: int = 3) -> str:
    """Pick the next drill: an unattempted one first, else your weakest
    (lowest make rate among drills with >= min_attempts), else the least-played."""
    stats = store.all_stats(drill_ids)
    unattempted = [d for d in drill_ids if stats[d].attempts == 0]
    if unattempted:
        return unattempted[0]
    established = [d for d in drill_ids if stats[d].attempts >= min_attempts]
    if established:
        return min(established, key=lambda d: stats[d].pct)
    return min(drill_ids, key=lambda d: stats[d].attempts)
