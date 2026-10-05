@echo off
rem Starts the Local PDF Redactor. First run sets everything up automatically.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating environment ^(a few minutes, needs internet once^)...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :error
    ".venv\Scripts\python.exe" -m spacy download en_core_web_sm || goto :error
    ".venv\Scripts\python.exe" -m spacy download fr_core_news_sm || goto :error
    ".venv\Scripts\python.exe" -m spacy download xx_ent_wiki_sm || goto :error
)
if not exist "tessdata\ara.traineddata" (
    ".venv\Scripts\python.exe" scripts\get_tessdata.py || goto :error
)

rem Streamlit asks for an email on first launch; an empty credentials file skips that prompt.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo Starting... your browser will open. Close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Something went wrong ^(see messages above^). Is Python 3.10+ installed and on PATH?
pause
exit /b 1
