@echo off
REM Main monitorability grid (resumable: re-run to continue after an interruption).
REM claude, qwen3-14b, gemma4 x A/B/C x 20 seeds + scripted controls, then both monitors, then analysis.
REM llama3.1-8b is left out: 0/12 side-task attempts across pilots 2-4.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist results\monitorability mkdir results\monitorability
set OUT=results\monitorability\main
set LOG=results\monitorability\main_log.txt
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; python -m monitorability.run_agents --agents greedy saboteur claude qwen3-14b gemma4 --conditions A B C --seeds 20 --out %OUT% 2>&1 | Tee-Object -FilePath %LOG% -Append; python -m monitorability.run_monitors --run %OUT% --monitors anthropic:claude-sonnet-4-6 ollama:qwen3:14b 2>&1 | Tee-Object -FilePath %LOG% -Append; python -m monitorability.analyze --run %OUT% 2>&1 | Tee-Object -FilePath %LOG% -Append"
echo.
echo GRID FINISHED. Results: %OUT%\FINDINGS_monitorability.md
pause
