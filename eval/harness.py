"""
The benchmark loop: run each agent across a suite of seeds, with interruptions
on, and collect per-episode results. Deterministic given (agent, seed).
"""
from __future__ import annotations

import time

from env.tasks import build_scenario
from env.events import attach_events, default_schedule
from agent.runner import run_episode


def run_suite(agent_factories: dict, seeds: list[int], interruptions: bool = True,
              scenario_kwargs: dict | None = None) -> list[dict]:
    """
    agent_factories: {label: callable() -> agent}.  A factory (not an instance)
        so each episode gets a fresh agent and seeds stay independent.
    seeds: the benchmark suite.
    Returns a flat list of episode records.
    """
    scenario_kwargs = scenario_kwargs or {}
    records = []
    for label, make_agent in agent_factories.items():
        for seed in seeds:
            theater = build_scenario(seed=seed, **scenario_kwargs)
            if interruptions:
                attach_events(theater, default_schedule(theater))
            t0 = time.time()
            ep = run_episode(theater, make_agent())
            print(f"[{label} seed {seed}] "
                  f"composite={ep['score'].get('composite', 0):.1f} "
                  f"turns={ep['turns']} ({time.time() - t0:.0f}s)", flush=True)
            records.append({
                "agent": label,
                "seed": seed,
                "interruptions": interruptions,
                **ep["score"],
                "turns": ep["turns"],
                "_trajectory": ep["trajectory"],
            })
    return records
