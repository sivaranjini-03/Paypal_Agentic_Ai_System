# Running the PayPal Agentic System

## Guaranteed browser/API demo

The project includes a dependency-light FastAPI web layer. It can run in **SAFE DEMO MODE** without PayPal credentials or the optional agent stack.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.web:app --host 127.0.0.1 --port 8000
```

Open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

In Swagger, call `POST /api/ask` with either:

```json
{"request":"What tools are available for managing invoices?"}
```

or:

```json
{"query":"What tools are available for managing invoices?"}
```

The response will explicitly say `mode: safe_demo` when optional agent dependencies are unavailable. No real PayPal operation is performed in demo mode.

## Full agent stack

The original full dependency list is preserved in `requirements-full.txt`. Install it when network/package access is available:

```powershell
pip install -r requirements-full.txt
```

Then configure `.env` and run the same FastAPI command. The API will use `mode: full_agent` when the full stack loads successfully.
