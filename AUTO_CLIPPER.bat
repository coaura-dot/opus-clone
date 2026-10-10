@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Abre a interface do Auto Clipper (janela com botoes: ligar/desligar e modo da GPU)
set PY=python
set PYW=pythonw
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if exist ".venv\Scripts\pythonw.exe" set PYW=.venv\Scripts\pythonw.exe
"%PY%" -c "import tkinter" >nul 2>&1
if errorlevel 1 goto semtk
start "" "%PYW%" interface.py
exit /b 0
:semtk
echo Este Python nao tem o tkinter, a parte de janelas.
echo Reinstale o Python pelo python.org marcando "tcl/tk and IDLE",
echo ou use INICIAR_AUTOMATICO.bat e AUTO_CLIPPER_MENU.bat.
pause
