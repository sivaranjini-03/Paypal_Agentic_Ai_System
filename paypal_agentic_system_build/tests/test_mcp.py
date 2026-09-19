"""Step 3 verification: dynamic MCP discovery and invocation."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.mcp.client import MCPClient
from app.mcp.server import DomainToolServer, mcp_name
from app.tools.credentials import CredentialManager
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

BASE_URL = "https://api.example.test"


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.from_collection()


def mock_executor(handler) -> ToolExecutor:
    settings = Settings(
        paypal_base_url=BASE_URL, paypal_client_id="cid", paypal_client_secret="secret"
    )

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 600})
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return ToolExecutor(
        settings=settings,
        credentials=CredentialManager(settings=settings, client=client),
        client=client,
    )


async def test_every_registry_tool_is_exposed_without_per_api_code(registry):
    server = DomainToolServer(registry)
    descriptors = server.list_tool_descriptors()
    assert len(descriptors) == len(registry)
    assert all(descriptor.input_schema["type"] == "object" for descriptor in descriptors)
    await server.aclose()


async def test_server_can_be_scoped_to_a_domain(registry):
    server = DomainToolServer(registry, domain="Disputes")
    descriptors = server.list_tool_descriptors()
    assert len(descriptors) == len(registry.get_tools_by_domain("Disputes"))
    assert all((d.meta or {})["domain"] == "Disputes" for d in descriptors)
    assert server.name == "tools-disputes"
    await server.aclose()

    with pytest.raises(ValueError):
        DomainToolServer(registry, domain="NoSuchDomain")


async def test_discovery_carries_workflow_metadata_and_no_credentials(registry):
    async with MCPClient.in_memory(domain="Payments") as client:
        tools = await client.list_tools()

    refund = next(t for t in tools if t.tool_id == "payments.refund_captured_payment")
    assert refund.name == "payments-refund_captured_payment"
    assert refund.requires == ["capture_id"]
    assert refund.produces == ["refund_id"]
    assert "capture_id" in refund.input_schema["properties"]

    blob = json.dumps([t.model_dump() for t in tools]).lower()
    assert "client_secret" not in blob and "bearer " not in blob


async def test_invocation_goes_through_the_executor(registry):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(201, json={"id": "REF-1", "status": "COMPLETED"})

    async with MCPClient.in_memory(
        domain="Payments", registry=registry, executor=mock_executor(handler)
    ) as client:
        result = await client.call_tool(
            "payments.refund_captured_payment", {"capture_id": "CAP-1", "body": {}}
        )

    assert result.success and result.status_code == 201
    assert seen["path"] == "/v2/payments/captures/CAP-1/refund"
    assert result.produced_context == {"refund_id": "REF-1"}


async def test_unknown_tool_returns_a_structured_error(registry):
    server = DomainToolServer(registry, domain="Payments")
    result = await server.call_tool("nope", {})
    assert result.is_error
    assert result.structured_content["error_type"] == "unknown_tool"
    await server.aclose()


async def test_new_collection_apis_are_discoverable_without_agent_changes(registry, tmp_path):
    """A registry gaining a tool immediately gains an MCP tool."""
    from app.tools.models import ToolDefinition
    from app.tools.registry import RegistryDocument

    extended = ToolRegistry(RegistryDocument.model_validate(registry.document.model_dump()))
    extended.register_tool(
        ToolDefinition(
            id="payments.brand_new_operation",
            name="Brand new operation",
            domain="Payments",
            hierarchy=["Payments"],
            method="GET",
            url_template="{{base_url}}/v2/payments/new",
            path_template="/v2/payments/new",
            source="test",
        )
    )
    server = DomainToolServer(extended, domain="Payments")
    names = {d.name for d in server.list_tool_descriptors()}
    assert mcp_name("payments.brand_new_operation") in names
    await server.aclose()


@pytest.mark.slow
async def test_real_mcp_session_over_stdio():
    """Proves protocol compliance end to end (spawns the server subprocess)."""
    async with MCPClient.stdio(domain="Disputes") as client:
        tools = await client.list_tools()
        assert tools and all(t.domain == "Disputes" for t in tools)
        assert any(t.tool_id == "disputes.list_disputes" for t in tools)
