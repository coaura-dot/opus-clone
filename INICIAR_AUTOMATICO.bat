@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
title Auto Clipper - Piloto Automatico
:loop
%PY% autopilot.py --auto
if %errorlevel%==0 goto fim
echo.
echo [!] O piloto caiu (codigo %errorlevel%). Religando em 60 segundos... (feche a janela pra parar)
timeout /t 60 /nobreak >nul
goto loop
:fim
pause
