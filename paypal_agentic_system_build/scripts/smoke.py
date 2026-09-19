"""Run the three mocked end-to-end smoke scenarios with UTF-8 console output."""

from __future__ import annotations

import asyncio
import sys

import httpx

from app.config import Settings
from app.system import build_system
from app.tools.registry import ToolRegistry


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def mock_paypal(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/oauth2/token":
        return httpx.Response(200, json={"access_token": "test-token", "expires_in": 600})
    if "/checkout/orders/" in path:
        return httpx.Response(
            200,
            json={
                "id": "ORDER-1",
                "purchase_units": [
                    {
                        "payments": {
                            "captures": [
                                {
                                    "id": "CAP-42",
                                    "status": "COMPLETED",
                                    "amount": {"value": "25.00", "currency_code": "USD"},
                                }
                            ]
                        }
                    }
                ],
            },
        )
    if path.endswith("/refund"):
        return httpx.Response(201, json={"id": "REF-42", "status": "COMPLETED"})
    if "disputes" in path:
        return httpx.Response(
            200, json={"items": [{"dispute_id": "PP-D-1", "status": "OPEN"}], "total_items": 1}
        )
    return httpx.Response(200, json={"id": "X", "status": "OK"})


async def main() -> int:
    configure_utf8_console()
    workflow = build_system(
        settings=Settings(
            paypal_base_url="https://api.example.test",
            paypal_client_id="test-client",
            paypal_client_secret="test-secret",
        ),
        registry=ToolRegistry.from_collection(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock_paypal)),
    )
    questions = [
        "Can I refund a payment that was only authorized?",
        "What tools are available for managing invoices?",
        "Find the payment for order ORDER-1 and refund it",
    ]
    try:
        for question in questions:
            state = await workflow.run(question)
            print(f"\nQ: {question}")
            print(
                f"   routed: {state.detected_domains} | "
                f"status: {state.workflow_status} | tokens: {state.total_tokens}"
            )
            print(f"   calls : {state.tool_calls}")
            print(f"   answer: {state.final_response[:230].replace(chr(10), ' ')}")
    finally:
        await workflow.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))