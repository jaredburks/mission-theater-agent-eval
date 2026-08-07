"""
Aggregation + failure-mode tagging. The aggregate table is the leaderboard;
the failure tags are the interesting half of the analysis.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean


def aggregate(records: list[dict]) -> dict:
    by_agent = defaultdict(list)
    for r in records:
        by_agent[r["agent"]].append(r)

    summary = {}
    for agent, rows in by_agent.items():
        summary[agent] = {
            "episodes": len(rows),
            "mean_composite": round(mean(r["composite"] for r in rows), 2),
            "solve_rate": round(mean(1.0 if r["all_secured"] else 0.0 for r in rows), 3),
            "mean_completion_pct": round(mean(r["completion_pct"] for r in rows), 1),
            "mean_steps": round(mean(r["steps_used"] for r in rows), 1),
            "mean_hazard_hits": round(mean(r["hazard_hits"] for r in rows), 2),
            "mean_units_lost": round(mean(r["units_lost"] for r in rows), 2),
        }
    return summary


def tag_failures(record: dict) -> list[str]:
    """Heuristic failure-mode labels for a single episode. The taxonomy is
    designed to grow; each tag is a named, checkable claim about a trajectory."""
    tags = []
    traj = record.get("_trajectory", [])
    if record["all_secured"]:
        return ["solved"]

    tools = [t["action"].get("tool") for t in traj]
    if not traj:
        return ["empty"]

    # gave up: hit budget with targets still unsecured but budget barely used
    if record["steps_used"] < 0.5 * record["budget"] and not record["all_secured"]:
        tags.append("early_termination")
    # reckless: lost a unit to a hazard
    if record["hazard_hits"] > 0:
        tags.append("hazard_death")
    # scan-starved: moved into unknown a lot without scanning
    if tools.count("scan") <= 1 and tools.count("move") >= 3:
        tags.append("under_scanned")
    # ignored interruption: an INTERRUPT fired but behavior never adapted
    if any(t["events"] for t in traj):
        idx = next(i for i, t in enumerate(traj) if t["events"])
        before = tools[max(0, idx - 2):idx]
        after = tools[idx:idx + 3]
        if before and after and before == after[:len(before)]:
            tags.append("ignored_interrupt")
    # looping: same action repeated with no state change
    if len(tools) >= 4 and len(set(tools[-4:])) == 1:
        tags.append("looping")
    return tags or ["incomplete_other"]


def failure_histogram(records: list[dict]) -> dict:
    hist = defaultdict(lambda: defaultdict(int))
    for r in records:
        for tag in tag_failures(r):
            hist[r["agent"]][tag] += 1
    return {a: dict(d) for a, d in hist.items()}
