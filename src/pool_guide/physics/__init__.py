"""Physics engines: turn a Shot into predicted ball Trajectories.

`simple` (numpy) is the default and runs everywhere. `pooltool` is an optional
higher-fidelity backend (needs Python <=3.12 + panda3d).
"""
from __future__ import annotations

from .engine import BallState, PhysicsEngine, Shot, Trajectory


def make_engine(cfg) -> PhysicsEngine:
    name = getattr(cfg, "engine", "simple")
    if name == "simple":
        from .simple import SimplePhysicsEngine
        return SimplePhysicsEngine(cfg)
    if name == "pooltool":
        from .pooltool_engine import PooltoolEngine
        return PooltoolEngine(cfg)
    raise ValueError(f"Unknown physics engine {name!r}")


__all__ = [
    "BallState", "PhysicsEngine", "Shot", "Trajectory", "make_engine",
]
