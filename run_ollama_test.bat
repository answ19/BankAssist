@echo off
REM One-click Ollama test for BankAssist (Windows). Double-click, or run from this folder.
REM Needs: Python 3 and Ollama installed and running.
cd /d "%~dp0"

echo === 1/5 Pulling qwen2.5:3b (skips if already downloaded)
ollama pull qwen2.5:3b || goto :fail

echo === 2/5 Setting up Python environment
if not exist .venv\Scripts\python.exe (py -3 -m venv .venv || python -m venv .venv || goto :fail)
.venv\Scripts\python.exe -m pip install -q -r requirements.txt || goto :fail

echo === 3/5 Running unit tests
.venv\Scripts\python.exe -m unittest discover -s tests || goto :fail

echo === 4/5 Training the baseline classifier if needed
if not exist models\baseline.joblib (.venv\Scripts\python.exe classifier.py || goto :fail)
.venv\Scripts\python.exe setup_database.py || goto :fail

echo === 5/5 Ollama tests (about 1-2 minutes)
.venv\Scripts\python.exe run_ollama_test.py > results_ollama_console.txt 2>&1
type results_ollama_console.txt
echo.
echo --- End-to-end examples with Ollama --- >> results_ollama_console.txt
for %%Q in ("I am still waiting on my new card" "how do I activate my card" "someone stole my card" "How long until my replacement card arrives?" "What is the exchange rate today?") do (
  echo. >> results_ollama_console.txt
  echo Q: %%~Q >> results_ollama_console.txt
  .venv\Scripts\python.exe pipeline.py --ollama "%%~Q" >> results_ollama_console.txt 2>&1
)
type results_ollama_console.txt
echo.
echo Done. Results: docs\ollama_answer_test.md, results\ollama_answer_test.csv, results_ollama_console.txt
pause
exit /b 0

:fail
echo.
echo A step failed. Check that Ollama is running (open the Ollama app) and Python is installed.
pause
exit /b 1
