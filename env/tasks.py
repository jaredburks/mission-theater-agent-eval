"""
Scenarios are generated deterministically from an integer seed, so a "suite"
is just a list of seeds. Same seed -> identical layout, every run. This is what
makes the benchmark reproducible.
"""
from __future__ import annotations

import random

from .theater import Theater, Unit, Target, DRONE, ROVER, UNIT_SPECS, chebyshev


def build_scenario(seed: int, width: int = 8, height: int = 8,
                   n_targets: int = 3, n_hazards: int = 6, n_walls: int = 5,
                   budget: int = 40) -> Theater:
    rng = random.Random(seed)
    all_cells = [(x, y) for x in range(width) for y in range(height)]
    rng.shuffle(all_cells)
    pool = iter(all_cells)

    def take(n):
        return [next(pool) for _ in range(n)]

    # Units start in a known "staging" corner so early game isn't blind luck.
    units = {
        "drone-1": Unit("drone-1", DRONE, (0, 0)),
        "rover-1": Unit("rover-1", ROVER, (0, 1)),
        "rover-2": Unit("rover-2", ROVER, (1, 0)),
    }
    occupied = {u.pos for u in units.values()}

    hazards, walls, targets = set(), set(), []
    for c in take(n_hazards):
        if c not in occupied:
            hazards.add(c)
    for c in take(n_walls):
        if c not in occupied and c not in hazards:
            walls.add(c)
    for c in take(n_targets):
        if c not in occupied and c not in hazards and c not in walls:
            targets.append(Target(pos=c, priority=1, status="hidden"))

    return Theater(
        width=width, height=height, walls=walls, hazards=hazards,
        units=units, targets=targets, budget=budget,
    )


def observe(theater: Theater) -> dict:
    """Fog-of-war observation handed to the agent each turn (NOT ground truth)."""
    known = theater.known
    return {
        "objective": theater.objective,
        "grid": {"width": theater.width, "height": theater.height},
        "step": theater.step_count,
        "budget_remaining": theater.budget - theater.step_count,
        "units": [
            {"id": u.id, "kind": u.kind, "pos": list(u.pos), "alive": u.alive,
             "move_range": UNIT_SPECS[u.kind]["move_range"],
             "scan_radius": UNIT_SPECS[u.kind]["scan_radius"],
             "can_secure": UNIT_SPECS[u.kind]["can_secure"]}
            for u in theater.units.values()
        ],
        "known_cells": sorted(map(list, known)),
        "known_hazards": sorted(list(c) for c in theater.hazards if c in known),
        "known_walls": sorted(list(c) for c in theater.walls if c in known),
        "located_targets": [list(t.pos) for t in theater.targets if t.status == "located"],
        "secured_count": sum(1 for t in theater.targets if t.status == "secured"),
        "target_total": len(theater.targets),
    }
