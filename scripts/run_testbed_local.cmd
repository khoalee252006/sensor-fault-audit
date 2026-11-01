@echo off
rem Study B full testbed on this laptop (double-click to start or resume; close the window to stop).
rem Every model is saved as soon as it is fitted, so stopping loses only the ~7 models in flight.
rem Progress: results\run_tb.out (one "[tb_...] rc=0" line per finished job; 261 jobs in total).
cd /d "%~dp0.."
start "Study B testbed" cmd /k ".venv\Scripts\python.exe scripts\run_all.py --suite testbed --jobs 8 --nj 2 --retries 1 >> results\run_tb.out 2>&1 & echo. & echo Finished - see results\run_tb.out"
