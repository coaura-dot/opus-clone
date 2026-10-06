@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
title Auto Clipper - Piloto Automatico
%PY% autopilot.py --auto
pause
