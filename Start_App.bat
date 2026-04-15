@echo off
title CCTV Face Reconstruction AI
echo ==========================================
echo Starting CCTV Face Reconstruction Server...
echo ==========================================

:: Change to the current directory of the script
cd /d "%~dp0"

:: Memory optimization — prevent OpenBLAS thread pool crashes
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

:: Open the browser
echo Opening browser to http://localhost:8000...
start http://localhost:8000

:: Run the server directly using virtual environment python
"venv\Scripts\python.exe" run.py

pause
