# PayPal Agentic System

An agent workflow that discovers PayPal API capabilities from a Postman
collection, routes a request by domain, plans tool calls, executes them through
MCP, and returns an evidence-based answer.

## Setup

1. Create `.env` from `.env.example` and set `GROQ_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the smoke scenarios: `.\.venv\Scripts\python.exe -m scripts.smoke`.

The smoke runner uses mocked PayPal HTTP responses, but it uses the configured
LLM for routing and planning. It configures UTF-8 output itself, so PowerShell
console encoding does not corrupt model output.

## Entry Points

- `python -m app.system "list open disputes"` runs one request from the CLI.
- `scripts/smoke.py` runs the knowledge, system-capability, and cross-domain
  refund scenarios.
- `pytest -q` runs the automated suite.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component flow.
## FastAPI browser interface

The project includes a FastAPI web interface.

Start it with:

```bash
python -m uvicorn app.web:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

- http://127.0.0.1:8000/ — browser interface
- http://127.0.0.1:8000/docs — FastAPI Swagger UI
- http://127.0.0.1:8000/redoc — ReDoc
- http://127.0.0.1:8000/health — health check

The browser page sends requests to `POST /api/ask` and displays the agent response, workflow status, domains, tools, token count, and errors.
