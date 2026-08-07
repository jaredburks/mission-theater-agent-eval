"""
The tests that matter for a verifiable environment: determinism and scoring
correctness. If these don't hold, the benchmark is meaningless.
Run: python -m pytest -q   (or: python tests/test_theater.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tasks import build_scenario, observe
from env.theater import ROVER


def test_scenario_is_deterministic():
    a = build_scenario(seed=7)
    b = build_scenario(seed=7)
    assert a.walls == b.walls
    assert a.hazards == b.hazards
    assert [t.pos for t in a.targets] == [t.pos for t in b.targets]


def test_different_seeds_differ():
    a = build_scenario(seed=1)
    b = build_scenario(seed=2)
    assert (a.hazards, [t.pos for t in a.targets]) != (b.hazards, [t.pos for t in b.targets])


def test_scan_locates_in_range():
    th = build_scenario(seed=3)
    u = next(iter(th.units.values()))
    # place a target adjacent to a unit and scan
    th.targets[0].pos = (u.pos[0] + 1, u.pos[1])
    th.targets[0].status = "hidden"
    th.step({"tool": "scan", "unit": u.id})
    assert th.targets[0].status in ("located", "secured")


def test_secure_requires_rover_on_cell():
    th = build_scenario(seed=4)
    rover = next(u for u in th.units.values() if u.kind == ROVER)
    th.targets[0].pos = rover.pos
    th.targets[0].status = "located"
    res = th.step({"tool": "secure", "unit": rover.id})
    assert res["ok"] and th.targets[0].status == "secured"


def test_score_is_pure_and_bounded():
    th = build_scenario(seed=5)
    s1 = th.score()
    s2 = th.score()
    assert s1 == s2
    assert s1["composite"] >= 0.0


def test_full_solve_beats_partial():
    th = build_scenario(seed=6)
    partial = th.score()["composite"]
    for t in th.targets:
        t.status = "secured"
    assert th.score()["composite"] > partial


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
