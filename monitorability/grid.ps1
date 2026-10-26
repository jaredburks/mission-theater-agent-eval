# Main grid as run for docs/MONITORABILITY.md: claude, qwen3-14b, gemma4 x A/B/C x 20 seeds (0-19)
# + scripted controls, then both monitors, then analysis. Resumable: re-run to continue.
param([string[]]$Agents = @("greedy", "saboteur", "claude", "qwen3-14b", "gemma4"),
      [int]$Seeds = 20)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$out = "results\monitorability\main"
python -m monitorability.run_agents --agents $Agents --conditions A B C --seeds $Seeds --out $out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m monitorability.run_monitors --run $out --monitors anthropic:claude-sonnet-4-6 ollama:qwen3:14b
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m monitorability.analyze --run $out
