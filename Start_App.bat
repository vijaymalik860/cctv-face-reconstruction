@echo off
title CCTV Face Reconstruction AI
echo ==========================================
echo Starting CCTV Face Reconstruction Server...
echo ==========================================

:: Change to the current directory
d:
cd "d:\cctv face reconstruction"

:: Open the browser
echo Opening browser to http://localhost:8000...
start http://localhost:8000

:: Activate Python virtual environment and run the server
call venv\Scripts\activate.bat
python run.py

pause
