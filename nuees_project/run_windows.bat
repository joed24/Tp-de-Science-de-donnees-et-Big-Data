@echo off
setlocal
cd /d "%~dp0"
python setup_environment.py
if errorlevel 1 exit /b %errorlevel%
python main.py
pause
