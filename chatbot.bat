@echo off
cd /d "%~dp0"
set VENV_PATH=.venv
call "%VENV_PATH%\Scripts\activate.bat"
start "" pythonw main_gui_styled.py