@echo off
rem Train a DraftNet model using the project venv.
rem
rem Usage:
rem   train FDN Premier
rem   train FDN Premier --top3
rem   train FDN Premier --top3 --embed
rem   train FDN Premier --top3 --embed --no-onnx
rem   train FDN Premier --keep-dataset
rem
rem All arguments after the mode are forwarded to run_training.py.

setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Error: venv not found at %PYTHON%
    echo Run: python -m venv venv ^&^& venv\Scripts\pip install -e .
    exit /b 1
)

"%PYTHON%" "%SCRIPT_DIR%run_training.py" %*
