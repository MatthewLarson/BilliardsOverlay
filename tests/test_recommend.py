"""Phase 4 tests: best-shot recommendation on the SimplePhysicsEngine."""
import time

import numpy as np

from pool_guide.config import PhysicsConfig
from pool_guide.physics import make_engine
from pool_guide.recommend import aim_to_pot, recommend_shot, standard_pockets

TABLE = (0.0, 0.0, 2540.0, 1270.0)
R = 57.15 / 2


def engine():
    # Coarser timestep for search: fast, and accurate enough for ranking. Step
    # size stays well under a ball diameter so there's no tunnelling.
    return make_engine(PhysicsConfig(dt=0.0035, max_time=8.0, sample_every=20))


def test_aim_to_pot_geometry():
    # Object at table centre, corner pocket at (2540,0). Ghost sits on the far
    # side of the object from the pocket.
    cue = (500, 635)
    target = (1270, 635)
    pocket = (2540, 0)
    aim, ghost, cut = aim_to_pot(cue, target, pocket, R)
    to_pocket = np.array(pocket) - np.array(target)
    to_pocket = to_pocket / np.linalg.norm(to_pocket)
    expected_ghost = np.array(target) - to_pocket * 2 * R
    assert np.allclose(ghost, expected_ghost, atol=1e-6)
    assert 0 <= cut <= 90


def test_recommends_a_makeable_pot():
    # Cue and object lined up straight at the right-middle... use a corner: object
    # just outside the top-right corner pocket, cue behind it on the pot line.
    pocket = (2540.0, 0.0)
    target = (2200.0, 300.0)
    # place cue on the line from pocket through target, further back
    d = np.array(target) - np.array(pocket)
    d = d / np.linalg.norm(d)
    cue = tuple(np.array(target) + d * 500)
    best = recommend_shot(cue, {"1": target}, TABLE, R, engine())
    assert best is not None
    assert best.target == "1"
    assert best.potted_target and not best.scratch
    # recommended pocket should be the corner we set up
    assert np.linalg.norm(np.array(best.pocket) - np.array(pocket)) < 1.0


def test_prefers_potting_over_hopeless():
    # Two targets: one easily pottable into a corner, one buried mid-table with no
    # clean line. The recommendation should be the makeable one and should pot.
    pocket = (0.0, 0.0)
    makeable = (300.0, 300.0)
    d = np.array(makeable) - np.array(pocket); d /= np.linalg.norm(d)
    cue = tuple(np.array(makeable) + d * 500)
    buried = (1270.0, 635.0)
    best = recommend_shot(cue, {"make": makeable, "buried": buried}, TABLE, R, engine())
    assert best is not None
    assert best.potted_target is True


def test_search_runtime_bounded():
    # A 6-ball spread; a full recommendation pass should be well under ~3s.
    balls = {str(i): (700 + 180 * i, 400 + 120 * (i % 3)) for i in range(6)}
    cue = (400.0, 635.0)
    t0 = time.perf_counter()
    best = recommend_shot(cue, balls, TABLE, R, engine())
    dt = time.perf_counter() - t0
    assert best is not None
    assert dt < 4.0, f"recommendation took {dt:.2f}s"


def test_top_n_returns_ranked_list():
    balls = {"1": (2200.0, 300.0), "2": (1500.0, 900.0)}
    cue = (500.0, 635.0)
    top = recommend_shot(cue, balls, TABLE, R, engine(), top_n=3)
    assert isinstance(top, list) and len(top) <= 3
    scores = [c.score for c in top]
    assert scores == sorted(scores, reverse=True)
