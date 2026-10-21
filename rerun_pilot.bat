@echo off
REM Re-run the pilot (resumes: only missing or errored episodes run), no git, no tests.
REM Double-click to run. Output is also saved to results\monitorability\pilot_park_rerun_log.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist results\monitorability mkdir results\monitorability
set LOG=results\monitorability\pilot_park_rerun_log.txt
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; python -m monitorability.run_agents --agents greedy saboteur qwen3-14b llama3.1-8b gemma4 claude --conditions A B C --seeds 2 --out results\monitorability\pilot_park 2>&1 | Tee-Object -FilePath %LOG%; python -m monitorability.run_monitors --run results\monitorability\pilot_park --monitors ollama:qwen3:14b anthropic:claude-sonnet-4-6 2>&1 | Tee-Object -FilePath %LOG% -Append; python -m monitorability.analyze --run results\monitorability\pilot_park 2>&1 | Tee-Object -FilePath %LOG% -Append"
echo.
echo PILOT FINISHED. Results: results\monitorability\pilot_park\FINDINGS_monitorability.md
pause
