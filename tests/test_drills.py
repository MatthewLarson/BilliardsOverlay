"""Phase 5 tests: drill library validity, session scoring, progress + suggestion."""
import numpy as np

from pool_guide.config import DrillsConfig
from pool_guide.drills import (
    DRILLS,
    DrillSession,
    Phase,
    ProgressStore,
    get_drill,
    list_drills,
    suggest_next,
)
from pool_guide.drills.model import to_mm
from pool_guide.drills.session import DetectedBall

TABLE = (0.0, 0.0, 2540.0, 1270.0)


def cfg():
    # small frame counts so scripted tests are short
    return DrillsConfig(stationary_frames=2, result_hold_frames=2, motion_thresh_mm=8.0)


def test_library_is_valid():
    assert DRILLS
    for d in list_drills():
        assert d.cue_spec is not None, f"{d.id} has no cue ball"
        for b in d.balls:
            assert 0.0 <= b.pos[0] <= 1.0 and 0.0 <= b.pos[1] <= 1.0
        for pk in d.target_pockets:
            assert 0 <= pk <= 5
        assert d.difficulty in range(1, 6)
        assert d.pots_needed() >= 0


def _still(session, ball_list, n):
    for _ in range(n):
        session.update(ball_list)


def _run_setup_ready(session, cue_mm, obj_mm):
    balls = [DetectedBall("cue", True, cue_mm), DetectedBall("o1", False, obj_mm)]
    # feed enough identical (stationary) frames to reach READY
    for _ in range(5):
        session.update(balls)
    return balls


def test_session_scores_a_made_pot():
    drill = get_drill("straight_pot")
    s = DrillSession(drill, TABLE, cfg())
    cue_mm = to_mm(drill.cue_spec.pos, TABLE)
    obj_mm = to_mm(drill.object_specs[0].pos, TABLE)

    _run_setup_ready(s, cue_mm, obj_mm)
    assert s.phase is Phase.READY

    # cue moves -> SHOOTING
    s.update([DetectedBall("cue", True, (cue_mm[0] + 50, cue_mm[1])),
              DetectedBall("o1", False, obj_mm)])
    assert s.phase is Phase.SHOOTING

    # object potted (gone), cue comes to rest; hold still -> RESULT + success
    rest = [DetectedBall("cue", True, (obj_mm[0] - 60, obj_mm[1]))]
    _still(s, rest, 3)
    assert s.phase is Phase.RESULT
    assert s.last_result.success and s.last_result.potted == 1
    assert s.makes == 1 and s.streak == 1


def test_session_scores_a_miss():
    drill = get_drill("straight_pot")
    s = DrillSession(drill, TABLE, cfg())
    cue_mm = to_mm(drill.cue_spec.pos, TABLE)
    obj_mm = to_mm(drill.object_specs[0].pos, TABLE)
    _run_setup_ready(s, cue_mm, obj_mm)

    s.update([DetectedBall("cue", True, (cue_mm[0] + 50, cue_mm[1])),
              DetectedBall("o1", False, obj_mm)])
    # object still on the table (not potted), both at rest
    rest = [DetectedBall("cue", True, (obj_mm[0] - 200, obj_mm[1])),
            DetectedBall("o1", False, (obj_mm[0] + 120, obj_mm[1] - 40))]
    _still(s, rest, 3)
    assert s.phase is Phase.RESULT
    assert not s.last_result.success and s.last_result.potted == 0


def test_session_detects_scratch():
    drill = get_drill("straight_pot")
    s = DrillSession(drill, TABLE, cfg())
    cue_mm = to_mm(drill.cue_spec.pos, TABLE)
    obj_mm = to_mm(drill.object_specs[0].pos, TABLE)
    _run_setup_ready(s, cue_mm, obj_mm)
    s.update([DetectedBall("cue", True, (cue_mm[0] + 50, cue_mm[1])),
              DetectedBall("o1", False, obj_mm)])
    # object potted but cue also gone (scratch) -> failure despite the pot
    _still(s, [], 3)
    assert s.phase is Phase.RESULT
    assert s.last_result.scratched and not s.last_result.success


def test_position_drill_requires_leave_zone():
    drill = get_drill("stop_shot")
    s = DrillSession(drill, TABLE, cfg())
    cue_mm = to_mm(drill.cue_spec.pos, TABLE)
    obj_mm = to_mm(drill.object_specs[0].pos, TABLE)
    zone_mm = to_mm(drill.cue_leave_zone.center, TABLE)
    _run_setup_ready(s, cue_mm, obj_mm)
    s.update([DetectedBall("cue", True, (cue_mm[0] + 50, cue_mm[1])),
              DetectedBall("o1", False, obj_mm)])
    # potted, but cue stops far from the leave zone -> should fail on position
    far = (zone_mm[0] + 600, zone_mm[1] + 400)
    _still(s, [DetectedBall("cue", True, far)], 3)
    assert not s.last_result.success
    assert "position" in s.last_result.reason


def test_progress_store_and_suggestion(tmp_path):
    path = tmp_path / "prog.json"
    store = ProgressStore(str(path))
    ids = ["a", "b", "c"]

    # unattempted -> suggest the first unplayed
    assert suggest_next(ids, store) == "a"

    # give a=weak, b=strong, c enough attempts
    for succ in [False, False, True]:
        store.record("a", succ)
    for succ in [True, True, True]:
        store.record("b", succ)
    for succ in [True, False, True]:
        store.record("c", succ)

    # persisted and reloadable
    store2 = ProgressStore(str(path))
    assert store2.stats("b").makes == 3
    assert store2.stats("a").pct < store2.stats("b").pct
    # weakest established drill is 'a' (0/... lowest pct)
    assert suggest_next(ids, store2) == "a"
