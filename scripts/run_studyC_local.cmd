@echo off
rem Study C (pre-registered, see docs/DEVIATIONS_C.md) on this laptop (double-click to start or resume; close the window to stop).
rem Every model is saved as soon as it is fitted, so stopping loses only the models in flight.
rem Progress: results\run_studyC.out (one "rc=0" line per finished job; 36 jobs, then 19_studyC_analyze.py).
cd /d "%~dp0.."
start "Study C testbed" cmd /k ".venv\Scripts\python.exe scripts\run_all.py --suite studyC --jobs 8 --nj 2 --retries 1 >> results\run_studyC.out 2>&1 & echo. & echo Finished - see results\run_studyC.out"
