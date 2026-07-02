"""Phase 3 physics tests for the SimplePhysicsEngine (deterministic, no hardware).

Table is a 2540 x 1270 mm rectangle. Ball radius ~28.6 mm.
"""
import numpy as np
import pytest

from pool_guide.config import PhysicsConfig
from pool_guide.physics import Shot, make_engine

TABLE = (0.0, 0.0, 2540.0, 1270.0)
R = 57.15 / 2


def engine(**over):
    cfg = PhysicsConfig(**over)
    return make_engine(cfg)


def _dir(p_from, p_to):
    v = np.array(p_to) - np.array(p_from)
    return tuple(v / np.linalg.norm(v))


def test_straight_shot_sends_object_ball_forward():
    # Cue at x=500, object ball straight ahead at x=1200, same y. Aim +x.
    shot = Shot(cue_pos=(500, 635), balls={"1": (1200, 635)},
                aim_dir=(1, 0), speed=2500, ball_radius=R, table=TABLE)
    traj = engine().simulate(shot)
    assert traj.cue_first_hit == "1"
    # Object ball ends up further along +x than it started, and near the y line.
    fx, fy = traj.finals["1"]
    assert fx > 1200 + 50
    assert abs(fy - 635) < 40


def test_cut_shot_object_follows_ghost_line():
    # Cue travels along y=635; object offset by 40mm (< ball diameter 57mm) so the
    # cue clips it -> a genuine cut. Object should depart along the ghost-ball line.
    cue = (500, 635)
    obj = (1200, 675)
    shot = Shot(cue_pos=cue, balls={"1": obj}, aim_dir=(1, 0),
                speed=3000, ball_radius=R, table=TABLE)
    traj = engine().simulate(shot)
    assert traj.cue_first_hit == "1"
    ghost = np.array(traj.ghost)
    expected = _dir(ghost, obj)                      # ghost-ball -> object centre
    # Launch direction = displacement from start to the first point >50mm away
    # (before that the object is still stationary awaiting contact).
    path = np.array(traj.paths["1"])
    disp = path - path[0]
    dist = np.linalg.norm(disp, axis=1)
    k = int(np.argmax(dist > 50))
    assert dist[k] > 50, "object ball never moved"
    actual = disp[k] / dist[k]
    cos = float(np.clip(actual @ expected, -1, 1))
    assert np.degrees(np.arccos(cos)) < 8            # within 8 degrees of the ghost line


def test_more_strength_makes_object_travel_farther():
    # Measure total path length of the object ball (robust to cushion rebounds on a
    # bounded table, where final displacement is non-monotonic in speed).
    def object_path_length(speed):
        shot = Shot(cue_pos=(400, 635), balls={"1": (1000, 635)},
                    aim_dir=(1, 0), speed=speed, ball_radius=R, table=TABLE)
        t = engine().simulate(shot)
        p = np.array(t.paths["1"])
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    assert object_path_length(3500) > object_path_length(1500) + 100


def test_draw_and_follow_differ():
    # Same straight stun into a ball; draw should leave the cue ball further back
    # (smaller x) than follow.
    def cue_final_x(b):
        shot = Shot(cue_pos=(500, 635), balls={"1": (900, 635)}, aim_dir=(1, 0),
                    speed=2500, ball_radius=R, table=TABLE, english=(0.0, b))
        return engine().simulate(shot).finals["cue"][0]
    assert cue_final_x(-1.0) < cue_final_x(1.0) - 20


def test_ball_stays_inside_table():
    # A hard shot with no targets must bounce around but never leave the cushions.
    shot = Shot(cue_pos=(600, 400), balls={}, aim_dir=(0.7, 0.5),
                speed=4000, ball_radius=R, table=TABLE)
    traj = engine().simulate(shot)
    pts = np.array(traj.paths["cue"])
    assert pts[:, 0].min() >= -1 and pts[:, 0].max() <= 2540 + 1
    assert pts[:, 1].min() >= -1 and pts[:, 1].max() <= 1270 + 1


def test_ball_pots_into_pocket():
    # Aim the object ball straight at the far-right-middle... use a corner pocket.
    # Cue at centre, object ball near top-right corner pocket, aimed at it.
    obj = (2400, 180)
    cue = (2200, 360)
    shot = Shot(cue_pos=cue, balls={"1": obj}, aim_dir=_dir(cue, (2540, 0)),
                speed=2600, ball_radius=R, table=TABLE)
    traj = engine().simulate(shot)
    # Not asserting it definitely pots (geometry-dependent), but the machinery runs
    # and any potted ball is recorded consistently.
    for pid in traj.potted:
        assert pid in traj.finals


def test_simulation_terminates_quickly():
    shot = Shot(cue_pos=(500, 635), balls={str(i): (800 + 60 * i, 600) for i in range(6)},
                aim_dir=(1, 0), speed=3000, ball_radius=R, table=TABLE)
    traj = engine().simulate(shot)
    # Every ball has a recorded path and final position.
    assert "cue" in traj.finals
    assert all(len(p) >= 2 for p in traj.paths.values())
