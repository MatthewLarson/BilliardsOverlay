"""Practice drills + coaching (Phase 5)."""
from .library import DRILLS, get_drill, list_drills
from .model import BallSpec, Drill, TargetZone
from .progress import ProgressStore, suggest_next
from .session import AttemptResult, DrillSession, Phase

__all__ = [
    "Drill", "BallSpec", "TargetZone",
    "DRILLS", "get_drill", "list_drills",
    "DrillSession", "AttemptResult", "Phase",
    "ProgressStore", "suggest_next",
]
