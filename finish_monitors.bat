@echo off
REM Fill in monitor calls that failed (e.g. out of API credit) and redo the analysis. Only missing calls run.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OUT=results\monitorability\main
set LOG=results\monitorability\main_log.txt
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; python -m monitorability.run_monitors --run %OUT% --monitors anthropic:claude-sonnet-4-6 ollama:qwen3:14b 2>&1 | Tee-Object -FilePath %LOG% -Append; python -m monitorability.analyze --run %OUT% 2>&1 | Tee-Object -FilePath %LOG% -Append"
echo.
echo MONITORS FINISHED. Results: %OUT%\FINDINGS_monitorability.md
pause
