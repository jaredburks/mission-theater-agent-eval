@echo off
REM Commit the monitorability work on its own branch, run the tests, then the pilot.
REM Double-click to run. Output is also saved to results\monitorability\pilot_park2_log.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
git switch monitorability
git add -A
git commit -q -m "Monitorability: per-turn side-task reminder in B and C; Sonnet monitor temperature fallback" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_014CyA4qbnKzVuFMAvZA89k7"
git log --oneline -1
python -m pip install -q pytest ollama anthropic matplotlib
python -m pytest -q
if not exist results\monitorability mkdir results\monitorability
set LOG=results\monitorability\pilot_park2_log.txt
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; python -m monitorability.run_agents --agents greedy saboteur qwen3-14b llama3.1-8b gemma4 claude --conditions A B C --seeds 2 --out results\monitorability\pilot_park2 2>&1 | Tee-Object -FilePath %LOG%; python -m monitorability.run_monitors --run results\monitorability\pilot_park2 --monitors ollama:qwen3:14b anthropic:claude-sonnet-4-6 2>&1 | Tee-Object -FilePath %LOG% -Append; python -m monitorability.analyze --run results\monitorability\pilot_park2 2>&1 | Tee-Object -FilePath %LOG% -Append"
echo.
echo PILOT FINISHED. Results: results\monitorability\pilot_park2\FINDINGS_monitorability.md
pause
