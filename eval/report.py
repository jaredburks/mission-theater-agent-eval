"""
Turns aggregate metrics + failure histogram into a markdown report (and an
optional bar chart if matplotlib is installed).

Each run is archived under results/trials/<timestamp>/ so re-runs don't clobber
history. results/report.md (+ plot) are refreshed as a convenient "latest" view.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from .metrics import aggregate, failure_histogram

TRIALS_ROOT = Path("results/trials")
LATEST_REPORT = Path("results/report.md")
LATEST_PLOT = Path("results/report_composite.png")


def new_trial_dir(root: str | Path = TRIALS_ROOT) -> Path:
    """Create and return a fresh timestamped trial directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / stamp
    if path.exists():
        path = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_report(records: list[dict], path: str | Path | None = None,
                 make_plot: bool = True, *,
                 trial_dir: str | Path | None = None,
                 update_latest: bool = True) -> str:
    """
    Write the markdown report (and optional plot).

    By default archives into a new results/trials/<timestamp>/ folder and also
    copies artifacts to results/report.md + results/report_composite.png.
    """
    if path is None:
        trial = Path(trial_dir) if trial_dir is not None else new_trial_dir()
        trial.mkdir(parents=True, exist_ok=True)
        path = trial / "report.md"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = aggregate(records)
    fails = failure_histogram(records)

    lines = ["# Mission Theater — Benchmark Report", ""]
    lines.append(f"Episodes: {len(records)}  |  Agents: {', '.join(summary)}")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    cols = ["agent", "mean_composite", "solve_rate", "mean_completion_pct",
            "mean_steps", "mean_hazard_hits", "mean_units_lost", "episodes"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for agent, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean_composite"]):
        lines.append("| " + " | ".join(str(s.get(c, agent) if c != "agent" else agent)
                                        for c in cols) + " |")

    lines.append("")
    lines.append("## Failure modes")
    lines.append("")
    for agent, hist in fails.items():
        pretty = ", ".join(f"{k}: {v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1]))
        lines.append(f"- **{agent}** — {pretty}")

    report = "\n".join(lines) + "\n"
    path.write_text(report, encoding="utf-8")
    print(f"wrote {path}")

    # Archive per-turn trajectories next to the report so failure modes can be
    # diagnosed after the fact (what did the agent actually do on seed N?).
    try:
        episodes = [{
            "agent": r["agent"], "seed": r["seed"],
            "score": {k: v for k, v in r.items()
                      if k not in ("_trajectory", "agent", "seed")},
            "trajectory": [{
                "turn": t["turn"],
                "action": t["action"],
                "result": t["result"],
                "events": t["events"],
            } for t in r.get("_trajectory", [])],
        } for r in records]
        traj_path = path.with_name("episodes.json")
        traj_path.write_text(json.dumps(episodes, indent=1, default=str),
                             encoding="utf-8")
        print(f"wrote {traj_path}")
    except Exception as e:
        print(f"(trajectory dump skipped: {e})")

    plot_path = path.with_name("report_composite.png")
    if make_plot:
        try:
            _plot(summary, plot_path)
        except Exception as e:  # matplotlib optional
            print(f"(plot skipped: {e})")
            plot_path = None
    else:
        plot_path = None

    if update_latest:
        _publish_latest(path, plot_path)

    return report


def _publish_latest(report_path: Path, plot_path: Path | None) -> None:
    """Refresh the top-level results/ artifacts as a pointer to the newest trial."""
    LATEST_REPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, LATEST_REPORT)
    print(f"updated {LATEST_REPORT}")
    if plot_path is not None and plot_path.exists():
        shutil.copy2(plot_path, LATEST_PLOT)
        print(f"updated {LATEST_PLOT}")


def _plot(summary: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = list(summary)
    vals = [summary[a]["mean_composite"] for a in agents]
    plt.figure(figsize=(6, 4))
    plt.bar(agents, vals)
    plt.ylabel("mean composite score")
    plt.title("Mission Theater — agent comparison")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"wrote {out}")
