@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.web:app --host 127.0.0.1 --port 8000
