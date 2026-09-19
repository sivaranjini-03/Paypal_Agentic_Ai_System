"""Step 7 verification: routing, cross-domain hand-off and aggregation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from app.agents.domain_agent import DomainAgent
from app.agents.host_agent import HostAgent
from app.config import Settings
from app.mcp.client import MCPClient
from app.tools.credentials import CredentialManager
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.retriever import ToolRetriever

BASE_URL = "https://api.example.test"


class ScriptedLLM:
    """Replies in order; each entry is the JSON payload for the next call."""

    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append("\n".join(content for _, content in messages))
        payload = self.responses.pop(0) if self.responses else {}
        return SimpleNamespace(
            content=json.dumps(payload),
            usage_metadata={"input_tokens": 50, "output_tokens": 10},
        )


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.from_collection()


@pytest.fixture(scope="module")
def retriever(registry) -> ToolRetriever:
    return ToolRetriever(registry, embedder=None)


def build_executor(handler) -> ToolExecutor:
    settings = Settings(
        paypal_base_url=BASE_URL, paypal_client_id="cid", paypal_client_secret="shh"
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


def build_host(registry, retriever, handler, llm) -> HostAgent:
    executor = build_executor(handler)

    def factory(domain: str) -> DomainAgent:
        client = MCPClient.in_memory(domain, registry=registry, executor=executor)
        agent = DomainAgent(domain, client=client, retriever=retriever, llm=llm)
        return agent

    host = HostAgent(registry=registry, retriever=retriever, llm=llm, agent_factory=factory)
    return host


# ---------------------------------------------------------------- routing ---
async def test_router_sees_domains_not_tools(registry, retriever):
    llm = ScriptedLLM(
        {"assignments": [{"domain": "Disputes", "objective": "find open disputes"}]}
    )
    host = build_host(registry, retriever, lambda r: httpx.Response(200, json={}), llm)

    calls = []
    decision = host.route("is there a dispute open from user_123", calls)

    assert decision.domains == ["Disputes"]
    prompt = llm.prompts[0]
    assert "Disputes" in prompt and "Invoices" in prompt
    assert "/v1/customer/disputes" not in prompt  # no endpoints in the routing prompt
    assert len(prompt) < 3000
    await host.aclose()


async def test_router_catalogue_describes_system_meta_questions(registry, retriever):
    llm = ScriptedLLM(
        {"assignments": [{"domain": "System", "objective": "list invoice capabilities"}]}
    )
    host = build_host(registry, retriever, lambda r: httpx.Response(200, json={}), llm)

    from app.capabilities import register_capabilities

    register_capabilities(registry)
    decision = host.route("what tools are available for managing invoices?", [])

    assert decision.domains == ["System"]
    prompt = llm.prompts[0]
    assert "Search system capabilities and activity" in prompt
    assert "capabilities" in prompt
    await host.aclose()


async def test_unknown_domains_from_the_model_are_dropped(registry, retriever):
    llm = ScriptedLLM(
        {
            "assignments": [
                {"domain": "Telepathy", "objective": "read minds"},
                {"domain": "Payments", "objective": "do the payment part"},
            ]
        }
    )
    host = build_host(registry, retriever, lambda r: httpx.Response(200, json={}), llm)

    decision = host.route("do a thing", [])
    assert decision.domains == ["Payments"]
    await host.aclose()


async def test_routing_falls_back_to_lexical_without_an_llm(registry, retriever):
    host = HostAgent(registry=registry, retriever=retriever, llm=None)
    host._llm_resolved = True
    decision = host.route("send an invoice for 50 dollars", [])
    assert "Invoices" in decision.domains
    await host.aclose()


async def test_unroutable_request_asks_for_clarification(registry, retriever):
    llm = ScriptedLLM({"assignments": [], "clarification": "That is not something I can do."})
    host = build_host(registry, retriever, lambda r: httpx.Response(200, json={}), llm)

    result = await host.handle("what is the weather in Madrid")
    assert result.status == "needs_input"
    assert "weather" in result.final_response.lower() or result.final_response
    assert not result.domain_results
    await host.aclose()


# ----------------------------------------------------------- cross-domain ---
async def test_cross_domain_workflow_passes_identifiers_downstream(registry, retriever):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if "/checkout/orders/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "id": "ORDER-1",
                    "purchase_units": [
                        {"payments": {"captures": [{"id": "CAP-42", "status": "COMPLETED"}]}}
                    ],
                },
            )
        return httpx.Response(201, json={"id": "REF-42", "status": "COMPLETED"})

    llm = ScriptedLLM(
        {
            "assignments": [
                {"domain": "Orders", "objective": "find the capture for the order"},
                {"domain": "Payments", "objective": "refund that capture"},
            ]
        },
        {
            "steps": [
                {
                    "step": 1,
                    "tool_id": "orders.show_order_details",
                    "inputs": {"order_id": "ORDER-1"},
                    "expects": ["capture_id"],
                }
            ]
        },
        {
            "steps": [
                {
                    "step": 1,
                    "tool_id": "payments.refund_captured_payment",
                    "inputs": {"capture_id": "{{capture_id}}", "body": {}},
                }
            ]
        },
        {"answer": "Refunded capture CAP-42 for order ORDER-1."},
    )
    host = build_host(registry, retriever, handler, llm)

    result = await host.handle("find the payment for order ORDER-1 and refund it")

    assert result.status == "success"
    assert result.detected_domains == ["Orders", "Payments"]
    assert seen == ["/v2/checkout/orders/ORDER-1", "/v2/payments/captures/CAP-42/refund"]
    assert result.context["capture_id"] == "CAP-42"
    assert result.context["refund_id"] == "REF-42"
    assert result.tool_calls == [
        "orders.show_order_details",
        "payments.refund_captured_payment",
    ]
    assert "CAP-42" in result.final_response
    await host.aclose()


async def test_failure_in_the_first_domain_stops_the_workflow(registry, retriever):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(404, json={"name": "RESOURCE_NOT_FOUND", "message": "no such order"})

    llm = ScriptedLLM(
        {
            "assignments": [
                {"domain": "Orders", "objective": "find the order"},
                {"domain": "Payments", "objective": "refund it"},
            ]
        },
        {
            "steps": [
                {
                    "step": 1,
                    "tool_id": "orders.show_order_details",
                    "inputs": {"order_id": "NOPE"},
                }
            ]
        },
    )
    host = build_host(registry, retriever, handler, llm)

    result = await host.handle("refund the payment for order NOPE")

    assert result.status == "recovery_required"
    assert len(result.domain_results) == 1  # Payments never ran
    assert len(calls) == 1
    assert "no such order" in result.errors[0]
    await host.aclose()


async def test_single_domain_requests_use_one_agent(registry, retriever):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [], "total_items": 0})

    llm = ScriptedLLM(
        {"assignments": [{"domain": "Disputes", "objective": "list open disputes"}]},
        {"steps": [{"step": 1, "tool_id": "disputes.list_disputes", "inputs": {}}]},
        {"answer": "You have no open disputes."},
    )
    host = build_host(registry, retriever, handler, llm)

    result = await host.handle("do I have any open disputes?")

    assert result.status == "success"
    assert len(result.domain_results) == 1
    assert result.final_response == "You have no open disputes."
    assert result.total_tokens > 0
    assert result.latency_ms > 0
    await host.aclose()


async def test_answer_is_synthesized_from_results_without_an_llm(registry, retriever):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": "REF-7", "status": "COMPLETED"})

    host = build_host(registry, retriever, handler, llm=None)
    host._llm_resolved = True
    result = await host.handle(
        "refund the captured payment", context={"capture_id": "CAP-7"}
    )

    assert result.status == "success"
    assert "refund_id" in result.final_response or "REF-7" in result.final_response
    await host.aclose()
