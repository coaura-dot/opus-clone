@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
rem modo da GPU: INICIAR_AUTOMATICO.bat 25 (ou 50, 70, 100). Sem numero: o
rem ultimo modo escolhido (fica salvo). Atalhos prontos: INICIAR_GPU_*.bat
set GPUARG=
if not "%~1"=="" set GPUARG=--gpu %~1
title Auto Clipper - Piloto Automatico
:loop
%PY% autopilot.py --auto %GPUARG%
if %errorlevel%==0 goto fim
echo.
echo [!] O piloto caiu (codigo %errorlevel%). Religando em 60 segundos... (feche a janela pra parar)
timeout /t 60 /nobreak >nul
goto loop
:fim
pause
