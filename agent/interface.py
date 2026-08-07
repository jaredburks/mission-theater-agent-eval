"""
The bridge between an LLM and the environment: the tool schema the model sees,
how an observation is rendered into a prompt, and how a model's reply is parsed
back into a validated action dict.
"""
from __future__ import annotations

import json
import re

# Anthropic-style tool definitions. (OpenAI/others: trivial reshape.)
TOOL_SCHEMA = [
    {
        "name": "move",
        "description": "Move a unit by (dx, dy). Drones move up to 2 cells (Chebyshev), "
                       "rovers up to 1. Moving into an UNSCANNED cell is risky: hazards "
                       "destroy the unit. Rovers cannot enter walls.",
        "input_schema": {
            "type": "object",
            "properties": {
                "unit": {"type": "string"},
                "dx": {"type": "integer"},
                "dy": {"type": "integer"},
            },
            "required": ["unit", "dx", "dy"],
        },
    },
    {
        "name": "scan",
        "description": "Reveal cells within the unit's scan radius and locate any "
                       "survivors in range. Always scan before moving into the unknown.",
        "input_schema": {
            "type": "object",
            "properties": {"unit": {"type": "string"}},
            "required": ["unit"],
        },
    },
    {
        "name": "secure",
        "description": "Secure a located survivor. The unit must be a ROVER standing on "
                       "the survivor's cell. This is the only action that scores completion.",
        "input_schema": {
            "type": "object",
            "properties": {"unit": {"type": "string"}},
            "required": ["unit"],
        },
    },
]

def to_openai_tools(schema=None):
    """Convert the Anthropic-style TOOL_SCHEMA into the OpenAI/Ollama 'tools' format.
    Ollama uses the OpenAI tool shape, so this adapter serves both."""
    schema = schema or TOOL_SCHEMA
    return [
        {"type": "function",
         "function": {"name": t["name"],
                      "description": t["description"],
                      "parameters": t["input_schema"]}}
        for t in schema
    ]


SYSTEM_PROMPT = """You are the operator of a small search-and-rescue fleet (drones + rovers).
Your objective: {objective}

Rules of thumb that good operators follow:
- You operate under FOG OF WAR. You only know cells you have scanned.
- SCAN before you MOVE into unknown territory. Unscanned hazards destroy units.
- Only ROVERS can SECURE survivors, and only while standing on the survivor's cell.
- You have a limited step budget. Each action consumes one step.
- Conditions change mid-mission. Watch for INTERRUPT messages and re-plan.

Each turn, call exactly one tool. Think briefly, then act."""


def render_observation(obs: dict, last_events: list[str] | None = None) -> str:
    lines = []
    if last_events:
        for e in last_events:
            lines.append(f"!! {e}")
    w, h = obs["grid"]["width"], obs["grid"]["height"]
    known = {tuple(c) for c in obs["known_cells"]}
    hazards = {tuple(c) for c in obs["known_hazards"]}
    walls = {tuple(c) for c in obs["known_walls"]}
    located = {tuple(c) for c in obs["located_targets"]}
    unit_at = {tuple(u["pos"]): u["id"] for u in obs["units"] if u["alive"]}

    lines.append(f"Step {obs['step']} | budget left: {obs['budget_remaining']} | "
                 f"secured {obs['secured_count']}/{obs['target_total']}")
    lines.append(f"Grid: {w}x{h} | known cells: {len(known)}/{w * h} "
                 f"(rescanning already-known cells reveals nothing — MOVE to explore)")
    for u in obs["units"]:
        status = "ALIVE" if u["alive"] else "LOST"
        lines.append(f"  {u['id']} ({u['kind']}) at {u['pos']} [{status}] "
                     f"move<= {u['move_range']} scan {u['scan_radius']} "
                     f"secure={u['can_secure']}")
    lines.append(f"Located survivors: {obs['located_targets'] or 'none yet'}")
    lines.append(f"Known hazards: {obs['known_hazards'] or 'none seen'}")
    lines.append(f"Known walls: {obs['known_walls'] or 'none seen'}")
    lines.append("Map (.=known empty, ?=unknown, H=hazard, W=wall, S=survivor, letter=unit):")
    # y=0 at top so the printed map matches grid coords reading down.
    for y in range(h):
        row = []
        for x in range(w):
            c = (x, y)
            if c in unit_at:
                row.append(unit_at[c][0])  # d/r
            elif c in located:
                row.append("S")
            elif c in hazards:
                row.append("H")
            elif c in walls:
                row.append("W")
            elif c in known:
                row.append(".")
            else:
                row.append("?")
        lines.append("  " + "".join(row))
    return "\n".join(lines)


_TOOL_NAMES = {"move", "scan", "secure", "noop"}


def parse_action(text_or_toolcall) -> dict:
    """Accepts either a structured tool-call dict {name, input} (preferred, from
    the Anthropic tool_use block) or a raw string the model emitted. Returns a
    validated action dict for Theater.step(). Always returns *something* (a noop
    on failure) so a malformed model turn costs a step rather than crashing the run.
    """
    # structured tool_use block
    if isinstance(text_or_toolcall, dict) and "name" in text_or_toolcall:
        name = text_or_toolcall["name"]
        args = text_or_toolcall.get("input", {}) or {}
        return _to_action(name, args)

    # raw string: find a JSON object
    if isinstance(text_or_toolcall, str):
        m = re.search(r"\{.*\}", text_or_toolcall, re.DOTALL)
        if m:
            try:
                blob = json.loads(m.group(0))
                name = blob.get("tool") or blob.get("name") or "noop"
                args = blob.get("input", blob)
                return _to_action(name, args)
            except json.JSONDecodeError:
                pass
    return {"tool": "noop"}


def _to_action(name: str, args: dict) -> dict:
    if name not in _TOOL_NAMES:
        return {"tool": "noop"}
    action = {"tool": name, "unit": args.get("unit")}
    if name == "move":
        action["dx"] = int(args.get("dx", 0))
        action["dy"] = int(args.get("dy", 0))
    return action
