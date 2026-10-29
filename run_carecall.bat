@echo off
title Follow-Up CareCall
cd /d "%~dp0"
echo === Follow-Up CareCall: build and start ===

rem 1. API keys: reuse the .env from the original project if this folder has none
if not exist .env (
  if exist "..\patient-chart-rag\.env" (
    copy "..\patient-chart-rag\.env" .env >nul
    echo Copied .env from patient-chart-rag
  ) else (
    echo No .env found. Copy .env.example to .env and add your keys, then run this again.
    pause & exit /b 1
  )
)

rem 2. Python: use this folder's environment, else reuse the original project's (same packages)
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=..\patient-chart-rag\.venv\Scripts\python.exe
if not exist %PY% (
  echo Creating a Python environment, this takes a few minutes...
  python -m venv .venv || (echo Python was not found. & pause & exit /b 1)
  set PY=.venv\Scripts\python.exe
  .venv\Scripts\python.exe -m pip install -r requirements-lock.txt || (pause & exit /b 1)
)
echo Using %PY%

rem 3. Build documents and chunks from the 200-patient CSVs
%PY% build_documents.py || (pause & exit /b 1)

rem 4. Upload to Pinecone once (skipped if already uploaded)
findstr /c:"\"uploaded\": true" data\processed\carecall_manifest.json >nul 2>&1
if errorlevel 1 (
  echo Uploading to Pinecone, 5 to 15 minutes the first time...
  %PY% ingest.py --upload || (pause & exit /b 1)
) else (
  echo Pinecone already loaded, skipping upload.
)

rem 5. Start the app (opens http://localhost:8503)
%PY% -m streamlit run app.py --server.port 8503
pause
