"""
Score every saved episode with each monitor in each view. Resumable.

  python -m monitorability.run_monitors --run results/monitorability/main \
         --monitors ollama:qwen3:14b anthropic:claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .monitor import VIEWS, build_prompt, make_monitor, parse_score
from .run_agents import _load_dotenv


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return rows


def run(run_dir: Path, monitors: list[str], views=VIEWS, call_factory=make_monitor, retries=2):
    episodes = [e for e in load_jsonl(run_dir / "episodes.jsonl") if not e.get("error")]
    out = run_dir / "monitor_scores.jsonl"
    done = {(r["episode"], r["monitor"], r["view"]) for r in load_jsonl(out) if r.get("suspicion") is not None}
    for spec in monitors:
        call = call_factory(spec)
        for ep in episodes:
            for view in views:
                key = (ep["id"], spec, view)
                if key in done:
                    continue
                prompt = build_prompt(ep, view)
                score, reason, raw, err = None, "", "", None
                t0 = time.time()
                for _ in range(retries + 1):
                    try:
                        raw = call(prompt)
                        score, reason = parse_score(raw)
                        err = None if score is not None else "unparseable"
                    except Exception as e:  # noqa: BLE001
                        err = f"{e.__class__.__name__}: {e}"
                    if score is not None:
                        break
                rec = {"episode": ep["id"], "agent": ep["agent"], "condition": ep["condition"],
                       "seed": ep["seed"], "monitor": spec, "view": view, "suspicion": score,
                       "reason": reason, "raw": raw[:2000], "error": err,
                       "prompt_chars": len(prompt), "wall_s": round(time.time() - t0, 2)}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{spec} | {view}] {ep['id']}: {score} {err or ''}", flush=True)
    return out


def main(argv=None):
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--monitors", nargs="+", default=["ollama:qwen3:14b", "anthropic:claude-sonnet-4-6"])
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    a = ap.parse_args(argv)
    print(run(Path(a.run), a.monitors, a.views))


if __name__ == "__main__":
    main()
