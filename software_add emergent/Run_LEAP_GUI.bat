
@echo off
REM --- Run_LEAP_GUI.bat ---
REM Double-click to launch your GUI with python. Adjust the path to your venv if needed.

setlocal

REM If you use a venv, uncomment and edit the next line:
REM call "%USERPROFILE%\miniconda3\Scripts\activate.bat" myenv

REM Change directory to this script's location (so relative imports work)
cd /d "%~dp0"

REM Launch the GUI
python main.py %*

endlocal
