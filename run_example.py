"""
Quickstart — runs the entire pipeline with NO API key, using scripted baselines.
    python run_example.py
Then drop in an LLM agent (see agent/models.py) and add it to AGENTS below.

Before a full LLM suite: python verify_models.py
"""
import os

from agent.baselines import RandomAgent, GreedyAgent
from agent.models import OllamaAgent, AnthropicAgent  # noqa: F401
from eval.harness import run_suite
from eval.report import write_report

def _load_dotenv(path=".env"):
    """Minimal .env loader (no dependency) so ANTHROPIC_API_KEY can live in a
    gitignored file instead of the shell environment."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
    except FileNotFoundError:
        pass


_load_dotenv()

CLAUDE_MODEL = os.environ.get("MT_CLAUDE_MODEL", "claude-sonnet-4-6")

# Factories (not instances) so each episode gets a fresh agent.
# Cross-family tool-callers (Meta / Alibaba / Google). think=False is the
# OllamaAgent default — required for stable Qwen/Gemma4 tool calls.
AGENTS = {
    "random":      lambda: RandomAgent(seed=0),
    "greedy":      lambda: GreedyAgent(),
    # ---- Local via Ollama (12-seed results archived in results/trials/) ----
    # "llama3.1-8b": lambda: OllamaAgent("llama3.1:8b", label="llama3.1-8b"),
    # "qwen3-14b":   lambda: OllamaAgent("qwen3:14b",   label="qwen3-14b"),
    # "gemma4":      lambda: OllamaAgent("gemma4:latest", label="gemma4"),
    # ---- Cloud (needs ANTHROPIC_API_KEY; run preflight_claude.py first) ----
    # "claude":      lambda: AnthropicAgent(CLAUDE_MODEL),
}

# MT_SEEDS=4 for a smoke run, 12 for the full suite.
SEEDS = list(range(int(os.environ.get("MT_SEEDS", "12"))))

if __name__ == "__main__":
    records = run_suite(AGENTS, SEEDS, interruptions=True)
    # Archives under results/trials/<timestamp>/; also refreshes results/report.*
    report = write_report(records, make_plot=True)
    print(report)
