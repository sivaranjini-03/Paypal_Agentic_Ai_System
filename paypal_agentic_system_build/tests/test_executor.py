"""Step 4 verification: generic executor, credential injection and schema building."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.tools.credentials import CredentialError, CredentialManager
from app.tools.executor import ToolExecutor, extract_context
from app.tools.registry import ToolRegistry
from app.tools.schema import build_input_schema

BASE_URL = "https://api.example.test"


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.from_collection()


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        paypal_base_url=BASE_URL,
        paypal_client_id="test-client",
        paypal_client_secret="test-secret",
    )


def make_executor(settings: Settings, handler) -> ToolExecutor:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    credentials = CredentialManager(settings=settings, client=client)
    return ToolExecutor(settings=settings, credentials=credentials, client=client)


def token_or(handler):
    """Wrap a handler so OAuth token requests are served automatically."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "tok-123", "expires_in": 3600})
        return handler(request)

    return _handler


# ------------------------------------------------------------------ schema --
def test_input_schema_exposes_parameters_but_not_credentials(registry):
    tool = registry.require_tool("payments.refund_captured_payment")
    schema = build_input_schema(tool)
    assert schema["properties"]["capture_id"]["type"] == "string"
    assert "capture_id" in schema["required"]
    assert "body" in schema["properties"]
    assert "Authorization" not in schema["properties"]
    assert "authorization" not in json.dumps(schema).lower()


# --------------------------------------------------------------- execution --
async def test_executes_a_parameterized_get(registry, settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"id": "CAP-1", "status": "COMPLETED"})

    executor = make_executor(settings, token_or(handler))
    tool = registry.require_tool("payments.show_captured_payment_details")
    result = await executor.execute(tool, {"capture_id": "CAP-1"})

    assert result.success and result.status_code == 200
    assert seen["url"] == f"{BASE_URL}/v2/payments/captures/CAP-1"
    assert seen["auth"] == "Bearer tok-123"
    assert result.data["status"] == "COMPLETED"
    await executor.aclose()


async def test_posts_a_body_and_extracts_produced_context(registry, settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        return httpx.Response(201, json={"id": "REF-9", "status": "COMPLETED"})

    executor = make_executor(settings, token_or(handler))
    tool = registry.require_tool("payments.refund_captured_payment")
    result = await executor.execute(
        tool, {"capture_id": "CAP-1", "body": {"amount": {"value": "10.00", "currency_code": "USD"}}}
    )

    assert result.success
    assert captured["path"] == "/v2/payments/captures/CAP-1/refund"
    assert captured["body"]["amount"]["value"] == "10.00"
    assert result.produced_context == {"refund_id": "REF-9"}
    await executor.aclose()


async def test_query_parameters_are_passed_through(registry, settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"transaction_details": []})

    executor = make_executor(settings, token_or(handler))
    tool = registry.require_tool("transaction_search.list_transactions")
    result = await executor.execute(
        tool, {"start_date": "2026-08-01T00:00:00-0700", "end_date": "2026-08-31T00:00:00-0700"}
    )

    assert result.success
    assert seen["params"]["start_date"] == "2026-08-01T00:00:00-0700"
    await executor.aclose()


async def test_missing_path_parameter_is_a_validation_failure(registry, settings):
    executor = make_executor(settings, token_or(lambda r: httpx.Response(200, json={})))
    tool = registry.require_tool("payments.refund_captured_payment")
    result = await executor.execute(tool, {})

    assert not result.success
    assert result.error_type == "validation"
    assert "capture_id" in result.error
    await executor.aclose()


async def test_http_errors_are_returned_structured_not_raised(registry, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "name": "UNPROCESSABLE_ENTITY",
                "message": "The requested action could not be performed.",
                "details": [{"issue": "CAPTURE_FULLY_REFUNDED"}],
            },
        )

    executor = make_executor(settings, token_or(handler))
    tool = registry.require_tool("payments.refund_captured_payment")
    result = await executor.execute(tool, {"capture_id": "CAP-1", "body": {}})

    assert not result.success
    assert result.status_code == 422
    assert "CAPTURE_FULLY_REFUNDED" in result.error
    await executor.aclose()


async def test_timeouts_are_classified(registry, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 600})
        raise httpx.ReadTimeout("too slow", request=request)

    executor = make_executor(settings, handler)
    tool = registry.require_tool("payments.show_captured_payment_details")
    result = await executor.execute(tool, {"capture_id": "CAP-1"})

    assert not result.success and result.error_type == "timeout"
    await executor.aclose()


async def test_rate_limit_headers_are_preserved_for_recovery(registry, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "too many"}, headers={"Retry-After": "2"})

    executor = make_executor(settings, token_or(handler))
    tool = registry.require_tool("payments.show_captured_payment_details")
    result = await executor.execute(tool, {"capture_id": "CAP-1"})

    assert result.status_code == 429
    assert result.response_headers["retry-after"] == "2"
    await executor.aclose()


# ------------------------------------------------------------- credentials --
async def test_token_is_cached_and_refreshable(settings):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"access_token": f"tok-{calls['count']}", "expires_in": 900})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = CredentialManager(settings=settings, client=client)

    assert await manager.get_access_token() == "tok-1"
    assert await manager.get_access_token() == "tok-1"
    assert calls["count"] == 1
    assert await manager.get_access_token(force_refresh=True) == "tok-2"
    await client.aclose()


async def test_missing_credentials_raise_a_credential_error():
    manager = CredentialManager(settings=Settings(paypal_client_id="", paypal_client_secret=""))
    with pytest.raises(CredentialError):
        manager.basic_auth_header()


async def test_basic_auth_tools_use_client_credentials(registry, settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"access_token": "x", "expires_in": 100})

    executor = make_executor(settings, handler)
    tool = registry.require_tool("authorization.generate_access_token")
    await executor.execute(tool, {})

    assert seen["auth"].startswith("Basic ")
    await executor.aclose()


# ------------------------------------------------------ context extraction --
def test_context_extraction_uses_declared_produces(registry):
    tool = registry.require_tool("orders.create_order")
    payload = {
        "id": "ORDER-1",
        "purchase_units": [{"payments": {"captures": [{"id": "CAP-1"}]}}],
        "capture_id": "CAP-1",
    }
    context = extract_context(tool, payload)
    assert context["order_id"] == "ORDER-1"
    assert context["capture_id"] == "CAP-1"
