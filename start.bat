@echo off
cd /d "%~dp0"
echo Starting AI Data Scientist...
echo.
echo  Open: http://127.0.0.1:8000
echo.
if not exist ".env" (
  echo  ERROR: .env file missing in this folder!
  echo  Copy .env.example to .env and add your ANTHROPIC_API_KEY
  pause
  exit /b 1
)
pip install python-dotenv anthropic -q 2>nul
python run_server.py
