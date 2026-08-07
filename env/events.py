"""
Interruptions are the whole point of "long-horizon judgment". They are injected
deterministically at a fixed step so they're reproducible, but the agent has no
warning. A good agent re-plans; a brittle one keeps executing its stale plan.

Attach to a Theater via attach_events(theater, schedule).
"""
from __future__ import annotations

from .theater import Theater, Target


def unit_offline(unit_id: str):
    def fn(th: Theater) -> str:
        u = th.units.get(unit_id)
        if u and u.alive:
            u.alive = False
            return f"INTERRUPT: {unit_id} has gone offline (comms lost)."
        return f"INTERRUPT: {unit_id} offline (already lost)."
    return fn


def new_priority_target(pos, priority: int = 3):
    def fn(th: Theater) -> str:
        th.targets.append(Target(pos=tuple(pos), priority=priority, status="hidden"))
        return (f"INTERRUPT: New HIGH-PRIORITY survivor reported near sector "
                f"{tuple(pos)} (priority {priority}). Not yet located.")
    return fn


def reprioritize(new_objective: str):
    def fn(th: Theater) -> str:
        th.objective = new_objective
        return f"INTERRUPT: Objective updated -> {new_objective}"
    return fn


def attach_events(theater: Theater, schedule: list[tuple[int, callable]]) -> Theater:
    """schedule = [(trigger_step, event_fn), ...]"""
    theater.events.extend(schedule)
    return theater


def default_schedule(theater: Theater) -> list[tuple[int, callable]]:
    """A reasonable interruption set scaled to the scenario."""
    mid = max(1, theater.budget // 3)
    far_corner = (theater.width - 1, theater.height - 1)
    return [
        (mid, unit_offline("rover-2")),
        (mid + 2, new_priority_target(far_corner, priority=3)),
    ]
