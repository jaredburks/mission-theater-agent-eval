"""
Runs one episode: agent <-> environment loop, returns a full trajectory plus the
programmatic score. The trajectory is the raw material for failure-mode analysis.
"""
from __future__ import annotations

from env.tasks import observe


def run_episode(theater, agent, max_turns: int | None = None) -> dict:
    max_turns = max_turns or theater.budget + 5
    last_events: list[str] = []
    trajectory = []

    turn = 0
    while not theater.is_done() and turn < max_turns:
        obs = observe(theater)
        action = agent.act(obs, last_events)
        result = theater.step(action)
        # Feed the step outcome back next turn so agents (esp. LLMs) can see that
        # a scan revealed nothing / a move landed / secure failed — not just interrupts.
        feedback = []
        if result.get("msg"):
            feedback.append(f"LAST RESULT: {result['msg']}")
        feedback.extend(result.get("events") or [])
        last_events = feedback
        trajectory.append({
            "turn": turn,
            "obs": obs,
            "action": action,
            "result": {k: result[k] for k in ("ok", "tool", "msg")},
            "events": result.get("events") or [],
        })
        turn += 1

    score = theater.score()
    return {
        "agent": getattr(agent, "name", agent.__class__.__name__),
        "score": score,
        "trajectory": trajectory,
        "turns": turn,
    }
