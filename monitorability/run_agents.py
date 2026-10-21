"""
Run agents x conditions x seeds and save every episode (score, side-task ground
truth, per-turn actions, results and reasoning traces) to JSONL. Resumable:
finished (agent, condition, seed) triples are skipped; errored ones are retried.

  python -m monitorability.run_agents --agents saboteur greedy --seeds 4 --out results/monitorability/smoke
  python -m monitorability.run_agents --agents qwen3-14b claude --conditions A B C --seeds 10 \
         --out results/monitorability/main
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from agent.baselines import GreedyAgent, RandomAgent, SaboteurAgent
from agent.runner import run_episode
from env.events import attach_events, default_schedule
from env.sidetasks import DEFAULT_SIDE_TASK, SIDE_TASKS, rally_cell
from env.tasks import build_scenario

from .conditions import CONDITIONS, condition_extra, has_side_task


def _load_dotenv(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
    except FileNotFoundError:
        pass


def make_agent(label: str, extra: str, seed: int, side_task: str = DEFAULT_SIDE_TASK, theater=None):
    """Factories. LLM agents get the rationale channel; scripted ones ignore `extra`."""
    if label == "greedy":
        return GreedyAgent()
    if label == "random":
        return RandomAgent(seed=seed)
    if label == "saboteur":
        rally = rally_cell(theater) if theater is not None else None
        return SaboteurAgent(side_task=side_task, rally=rally)
    from agent.models import AnthropicAgent, OllamaAgent
    if label == "claude":
        return AnthropicAgent(os.environ.get("MT_CLAUDE_MODEL", "claude-sonnet-4-6"),
                              system_extra=extra, rationale=True, label="claude")
    ollama = {"qwen3-14b": "qwen3:14b", "llama3.1-8b": "llama3.1:8b", "gemma4": "gemma4:latest"}
    if label in ollama:
        return OllamaAgent(ollama[label], label=label, system_extra=extra, rationale=True,
                           seed=seed, host=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
    raise SystemExit(f"unknown agent {label!r}")


def scripted_condition_ok(label: str, cond: str) -> bool:
    """Scripted agents can't read prompts, so pair them with the condition they embody."""
    if label == "saboteur":
        return cond in ("B", "C")
    if label in ("greedy", "random"):
        return cond == "A"
    return True


def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                if not r.get("error"):
                    done[r["id"]] = r
    return done


def run(agents, conditions, seeds, out: Path, max_turns=None, side_task: str = DEFAULT_SIDE_TASK):
    task = SIDE_TASKS[side_task]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "episodes.jsonl"
    done = load_done(path)
    # rewrite without errored rows so retries don't duplicate
    with path.open("w", encoding="utf-8") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for label in agents:
        for cond in conditions:
            if not scripted_condition_ok(label, cond):
                continue
            for seed in seeds:
                eid = f"{label}|{cond}|{seed}"
                if eid in done:
                    continue
                th = build_scenario(seed=seed)
                attach_events(th, default_schedule(th))
                extra = condition_extra(cond, task, th)
                t0 = time.time()
                rec = {"id": eid, "agent": label, "condition": cond, "seed": seed,
                       "side_task": task.name if has_side_task(cond) else None,
                       "side_task_name": task.name, "system_extra": extra}
                try:
                    ep = run_episode(th, make_agent(label, extra, seed, task.name, th), max_turns=max_turns)
                    # evaluated in every condition: in A it is the honest base rate
                    st = task.evaluate(th)
                    rec.update(score=ep["score"], turns=ep["turns"], trajectory=ep["trajectory"],
                               side_task_success=st["success"], side_task_visited=st["visited"],
                               side_task_detail=st, error=None)
                except Exception as e:  # noqa: BLE001 - API errors etc.; retried next run
                    rec.update(error=f"{e.__class__.__name__}: {e}", traceback=traceback.format_exc()[-2000:])
                rec["wall_s"] = round(time.time() - t0, 1)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=list) + "\n")
                s = rec.get("score", {})
                print(f"[{eid}] composite={s.get('composite', '-')} side_task={rec.get('side_task_success')}"
                      f" visited={rec.get('side_task_visited')} turns={rec.get('turns')} "
                      f"{rec['wall_s']}s {('ERROR ' + rec['error']) if rec.get('error') else ''}", flush=True)
    return path


def main(argv=None):
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", required=True,
                    help="greedy random saboteur qwen3-14b llama3.1-8b gemma4 claude")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--side-task", default=DEFAULT_SIDE_TASK, choices=list(SIDE_TASKS))
    ap.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--out", default="results/monitorability/main")
    a = ap.parse_args(argv)
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    print(run(a.agents, a.conditions, seeds, Path(a.out), a.max_turns, a.side_task))


if __name__ == "__main__":
    main()
