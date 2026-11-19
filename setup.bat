@echo off
rem One-time setup: create the Python environment on this drive and install everything.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Could not create the environment. Install Python 3.10-3.12 from python.org first.
  pause
  exit /b 1
)
rem keep pip downloads in this folder instead of C:
if not exist ".venv\pip.ini" (echo [global]& echo cache-dir = %~dp0.cache\pip) > ".venv\pip.ini"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m gesturebridge.check
pause
