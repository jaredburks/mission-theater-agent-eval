@echo off
REM Commit the monitorability work on its own branch, run the tests, then the pilot.
REM Double-click to run. Output is also saved to results\monitorability\pilot_log.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
git switch monitorability 2>nul || git switch -c monitorability
git add -A
git commit -q -m "Monitorability red-team: covert side task, rationale channel, monitors, analysis" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_014CyA4qbnKzVuFMAvZA89k7"
git log --oneline -1
python -m pip install -q pytest ollama anthropic matplotlib
python -m pytest -q
if not exist results\monitorability mkdir results\monitorability
powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\monitorability\pilot.ps1 2>&1 | Tee-Object -FilePath results\monitorability\pilot_log.txt"
echo.
echo PILOT FINISHED. Results: results\monitorability\pilot\FINDINGS_monitorability.md
pause
