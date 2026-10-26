# Pilot: every LLM agent x A/B/C x 2 seeds, then the cheap local monitor and analysis.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$out = "results\monitorability\pilot_rt"
python -m monitorability.run_agents --agents greedy saboteur qwen3-14b llama3.1-8b gemma4 claude `
    --conditions A B C --seeds 2 --out $out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m monitorability.run_monitors --run $out --monitors ollama:qwen3:14b anthropic:claude-sonnet-4-6
python -m monitorability.analyze --run $out
