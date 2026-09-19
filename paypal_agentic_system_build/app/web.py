"""FastAPI web interface for the PayPal Agentic System.

The web layer is intentionally dependency-light: the full agent stack is loaded
only when an /api/ask request is made. If optional agent dependencies are not
installed, the browser demo still runs in SAFE DEMO MODE so the UI and API can
be verified without credentials or network access.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AgentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    request: str = Field(min_length=1, alias="query")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.workflow = None
    yield
    workflow = getattr(app.state, "workflow", None)
    if workflow is not None:
        await workflow.aclose()


app = FastAPI(
    title="PayPal Agentic System",
    description="FastAPI web interface for the PayPal agentic workflow.",
    version="1.1.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))


def demo_response(user_request: str) -> dict[str, Any]:
    """Return a deterministic response when optional agent dependencies are absent."""
    text = user_request.lower()
    sensitive = any(word in text for word in ("refund", "capture", "void", "cancel"))
    if "invoice" in text:
        domain = "Invoices"
        tool = "invoice capability discovery"
    elif "dispute" in text:
        domain = "Disputes"
        tool = "dispute capability discovery"
    elif any(word in text for word in ("refund", "payment", "capture", "order")):
        domain = "Payments"
        tool = "payment capability discovery"
    else:
        domain = "General"
        tool = "tool capability discovery"

    status = "confirmation_required" if sensitive else "demo_ready"
    answer = (
        f"SAFE DEMO MODE: understood the request as a {domain} operation. "
        f"The architecture would retrieve the appropriate tool ({tool}), "
        "create an execution plan, and route execution through MCP. "
        + ("A user confirmation is required before this sensitive operation." if sensitive else "No external PayPal API call was made.")
    )
    return {
        "answer": answer,
        "status": status,
        "domains": [domain],
        "tools": [tool],
        "tokens": 0,
        "errors": [],
        "mode": "safe_demo",
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "PayPal Agentic System"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "paypal-agentic-system", "mode": "web"}


@app.post("/api/ask")
async def ask(payload: AgentRequest) -> dict[str, Any]:
    # Keep the web application runnable even when the optional full agent stack
    # (FastMCP, MCP, LangGraph, SentenceTransformers, etc.) is not installed.
    try:
        from app.system import build_system
    except (ImportError, ModuleNotFoundError):
        return demo_response(payload.request)

    workflow = build_system()
    try:
        state = await workflow.run(payload.request)
        return {
            "answer": state.final_response,
            "status": state.workflow_status,
            "domains": state.detected_domains,
            "tools": state.tool_calls,
            "tokens": state.total_tokens,
            "errors": state.errors,
            "mode": "full_agent",
        }
    except (ImportError, ModuleNotFoundError) as exc:
        return {**demo_response(payload.request), "errors": [str(exc)]}
    finally:
        await workflow.aclose()


@app.get("/api/info")
async def info() -> dict[str, Any]:
    return {
        "name": "PayPal Agentic System",
        "framework": "FastAPI",
        "docs": "/docs",
        "health": "/health",
        "ask_endpoint": "/api/ask",
        "safe_demo_mode": True,
    }
