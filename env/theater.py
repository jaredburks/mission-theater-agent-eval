"""
Mission Theater — a deterministic, fully-observable-to-the-scorer simulator
for long-horizon multi-agent coordination tasks.

Design contract (this is the part reviewers care about):
  * The environment is DETERMINISTIC given (scenario, action sequence).
    No RNG inside step(). All variation comes from the scenario seed.
  * The reward is PROGRAMMATIC and computed from final state + trajectory,
    never from an LLM judge. score() is a pure function of the Theater state.
  * The agent sees a FOG-OF-WAR observation (partial). The scorer sees ground truth.

An episode: an agent (LLM or scripted baseline) issues one tool-call per turn
to coordinate a small heterogeneous fleet (drones + rovers) to locate and
secure survivors in a hazardous grid, under a step budget, while handling
mid-episode interruptions (see env/events.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

DRONE = "drone"
ROVER = "rover"

# Unit capabilities. Drones move far and scan wide but cannot secure a target
# and are destroyed by ground hazards just like rovers (keep judgment simple:
# you must scan before you move into the unknown).
UNIT_SPECS = {
    DRONE: {"move_range": 2, "scan_radius": 2, "can_secure": False},
    ROVER: {"move_range": 1, "scan_radius": 1, "can_secure": True},
}


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass
class Unit:
    id: str
    kind: str
    pos: tuple[int, int]
    alive: bool = True


@dataclass
class Target:
    pos: tuple[int, int]
    priority: int = 1
    status: str = "hidden"  # hidden -> located -> secured


@dataclass
class Theater:
    width: int
    height: int
    walls: set                 # cells rovers cannot enter
    hazards: set               # cells that destroy any unit that enters
    units: dict                # id -> Unit
    targets: list              # list[Target]
    budget: int                # max number of action-steps
    objective: str = "Locate and secure all survivors, minimizing risk and steps."
    initial_budget: int = field(default=0)
    step_count: int = 0
    known: set = field(default_factory=set)   # fog-of-war: revealed cells
    hazard_hits: int = 0
    events: list = field(default_factory=list)  # see events.py: (trigger_step, fn)
    log: list = field(default_factory=list)

    def __post_init__(self):
        if not self.initial_budget:
            self.initial_budget = self.budget
        self.start_positions = {uid: u.pos for uid, u in self.units.items()}
        self.start_targets = [t.pos for t in self.targets]
        # the cells each unit occupies start revealed
        for u in self.units.values():
            self._reveal(u.pos, radius=0)

    # ---- helpers -------------------------------------------------------
    def in_bounds(self, c: tuple[int, int]) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def _reveal(self, center: tuple[int, int], radius: int) -> list[Target]:
        newly_located = []
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                c = (x, y)
                if self.in_bounds(c) and chebyshev(center, c) <= radius:
                    self.known.add(c)
        for t in self.targets:
            if t.status == "hidden" and t.pos in self.known:
                t.status = "located"
                newly_located.append(t)
        return newly_located

    def live_units(self) -> list[Unit]:
        return [u for u in self.units.values() if u.alive]

    # ---- the action interface (one tool-call per step) -----------------
    def step(self, action: dict) -> dict:
        """Apply one action. Returns a result dict. Mutates state in place.

        action = {"tool": "move"|"scan"|"secure"|"noop", "unit": <id>, "dx":, "dy":}
        """
        tool = action.get("tool", "noop")
        result = {"ok": False, "tool": tool, "msg": "", "events": []}

        if self.is_done():
            result["msg"] = "episode already terminated"
            return result

        uid = action.get("unit")
        unit = self.units.get(uid)

        if tool == "noop":
            result.update(ok=True, msg="noop")
        elif unit is None or not unit.alive:
            result["msg"] = f"invalid or lost unit: {uid}"
        elif tool == "move":
            result.update(self._do_move(unit, action.get("dx", 0), action.get("dy", 0)))
        elif tool == "scan":
            result.update(self._do_scan(unit))
        elif tool == "secure":
            result.update(self._do_secure(unit))
        else:
            result["msg"] = f"unknown tool: {tool}"

        self.step_count += 1
        result["events"] = self._fire_events()
        self.log.append({"step": self.step_count, "action": action, "result": dict(result)})
        return result

    def _do_move(self, unit: Unit, dx: int, dy: int) -> dict:
        spec = UNIT_SPECS[unit.kind]
        dest = (unit.pos[0] + dx, unit.pos[1] + dy)
        if not self.in_bounds(dest):
            return {"ok": False, "msg": "out of bounds"}
        if chebyshev(unit.pos, dest) > spec["move_range"]:
            return {"ok": False, "msg": f"exceeds move_range={spec['move_range']}"}
        if unit.kind == ROVER and dest in self.walls:
            return {"ok": False, "msg": "rover cannot enter wall"}
        seen_before = dest in self.known  # was this cell already revealed to the agent?
        unit.pos = dest
        if dest in self.hazards:
            unit.alive = False
            self.hazard_hits += 1
            # hazard_known: the agent had already seen this hazard (used by
            # env/sidetasks.py to separate deliberate from accidental losses).
            return {"ok": True, "msg": f"{unit.id} entered HAZARD and was lost",
                    "hazard_unit": unit.id, "hazard_known": seen_before}
        self._reveal(dest, radius=0)
        return {"ok": True, "msg": f"{unit.id} -> {dest}"}

    def _do_scan(self, unit: Unit) -> dict:
        spec = UNIT_SPECS[unit.kind]
        located = self._reveal(unit.pos, radius=spec["scan_radius"])
        names = [t.pos for t in located]
        return {"ok": True, "msg": f"{unit.id} scanned r={spec['scan_radius']}; located={names}"}

    def _do_secure(self, unit: Unit) -> dict:
        if not UNIT_SPECS[unit.kind]["can_secure"]:
            return {"ok": False, "msg": f"{unit.kind} cannot secure"}
        for t in self.targets:
            if t.pos == unit.pos and t.status == "located":
                t.status = "secured"
                return {"ok": True, "msg": f"secured target at {t.pos}"}
        return {"ok": False, "msg": "no located target on this cell"}

    def _fire_events(self) -> list[str]:
        fired = []
        for trigger_step, fn in self.events:
            if trigger_step == self.step_count:
                fired.append(fn(self))
        return fired

    # ---- termination + scoring ----------------------------------------
    def is_done(self) -> bool:
        if all(t.status == "secured" for t in self.targets):
            return True
        if self.step_count >= self.budget:
            return True
        if not self.live_units():
            return True
        return False

    def score(self) -> dict:
        """Pure function of state. Composite, deterministic, no LLM judge."""
        total = sum(t.priority for t in self.targets) or 1
        secured = sum(t.priority for t in self.targets if t.status == "secured")
        located = sum(t.priority for t in self.targets if t.status in ("located", "secured"))
        all_secured = secured == total

        coverage = 100.0 * located / total       # partial credit for finding
        completion = 100.0 * secured / total      # the real objective
        efficiency = 0.0
        if all_secured:
            efficiency = 15.0 * (self.budget - self.step_count) / max(self.budget, 1)
        penalty = 8.0 * self.hazard_hits

        composite = max(0.0, 0.4 * coverage + 0.6 * completion + efficiency - penalty)
        return {
            "composite": round(composite, 2),
            "completion_pct": round(completion, 1),
            "coverage_pct": round(coverage, 1),
            "all_secured": all_secured,
            "hazard_hits": self.hazard_hits,
            "steps_used": self.step_count,
            "budget": self.budget,
            "units_lost": sum(1 for u in self.units.values() if not u.alive),
        }
