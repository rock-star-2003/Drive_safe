@echo off
title Driver Monitoring System Setup

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install Python.
    pause
    exit /b
)

:: Create virtual environment if not exists
if not exist "env\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv env
)

:: Activate environment
call env\Scripts\activate.bat

:: Install packages if not already installed
pip show opencv-python >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install opencv-python mediapipe numpy pyttsx3 Pillow
)

:: Run the app if script exists
if exist "driver_monitoring_tkinter.py" (
    echo.
    echo Installation complete!
    echo Press any key to run the application...
    pause
    python driver_monitoring_tkinter.py
) else (
    echo tkinter.py not found!
    pause
)
