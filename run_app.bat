@echo off
rem Start the GestureBridge app and open it in the browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://127.0.0.1:8502"
".venv\Scripts\python.exe" -m streamlit run app.py
