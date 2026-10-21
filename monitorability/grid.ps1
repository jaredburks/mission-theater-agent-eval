# Main grid: 4 LLM agents x A/B/C x 10 seeds (seeds 0-9) + scripted controls. Resumable.
param([string[]]$Agents = @("greedy", "saboteur", "claude", "qwen3-14b", "gemma4", "llama3.1-8b"),
      [int]$Seeds = 10)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$out = "results\monitorability\main"
python -m monitorability.run_agents --agents $Agents --conditions A B C --seeds $Seeds --out $out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m monitorability.analyze --run $out
