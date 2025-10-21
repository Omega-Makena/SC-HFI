@echo off
echo ========================================
echo   SF-HFE v2.0 - Dashboard
echo ========================================
echo.

REM Check if venv exists
if not exist "..\venv\" (
    echo Creating virtual environment...
    cd ..
    python -m venv venv
    cd sf_hfe_v2
    echo.
    echo Installing dependencies...
    ..\venv\Scripts\pip.exe install torch numpy matplotlib streamlit pandas
    echo.
)

echo Starting dashboard...
echo Open browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop
echo.

..\venv\Scripts\streamlit.exe run dashboard.py

pause

