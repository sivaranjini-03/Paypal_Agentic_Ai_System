"""Step 6 verification: domain agent planning, execution and result inspection."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from app.agents.domain_agent import DomainAgent
from app.agents.models import DomainAgentInput
from app.config import Settings
from app.mcp.client import MCPClient
from app.tools.credentials import CredentialManager
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.retriever import ToolRetriever

BASE_URL = "https://api.example.test"


class FakeLLM:
    """Returns scripted JSON completions and reports token usage."""

    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append("\n".join(content for _, content in messages))
        payload = self.responses.pop(0) if self.responses else {"steps": []}
        return SimpleNamespace(
            content=json.dumps(payload),
            usage_metadata={"input_tokens": 100, "output_tokens": 20},
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
        settings=settings, credentials=CredentialManager(settings=settings, client=client),
        client=client,
    )


async def build_agent(domain, registry, retriever, handler, llm=None) -> DomainAgent:
    client = await MCPClient.in_memory(
        domain, registry=registry, executor=build_executor(handler)
    ).connect()
    return DomainAgent(domain, client=client, retriever=retriever, llm=llm)


# ---------------------------------------------------------------- planning --
async def test_single_step_plan_executes_and_returns_context(registry, retriever):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(201, json={"id": "REF-1", "status": "COMPLETED"})

    llm = FakeLLM(
        {
            "reasoning": "refund the capture",
            "steps": [
                {
                    "step": 1,
                    "purpose": "refund",
                    "tool_id": "payments.refund_captured_payment",
                    "inputs": {"capture_id": "CAP-9", "body": {}},
                    "expects": ["refund_id"],
                }
            ],
        }
    )
    agent = await build_agent("Payments", registry, retriever, handler, llm)
    result = await agent.run(DomainAgentInput(user_request="refund capture CAP-9", domain="Payments"))

    assert result.status == "success"
    assert calls == ["/v2/payments/captures/CAP-9/refund"]
    assert result.context["refund_id"] == "REF-1"
    assert result.total_tokens == 120
    await agent.aclose()


async def test_multi_step_plan_passes_values_between_steps(registry, retriever):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/capture"):
            return httpx.Response(201, json={"id": "CAP-77", "status": "COMPLETED"})
        return httpx.Response(201, json={"id": "REF-77", "status": "COMPLETED"})

    llm = FakeLLM(
        {
            "reasoning": "capture then refund",
            "steps": [
                {
                    "step": 1,
                    "purpose": "capture the authorization",
                    "tool_id": "payments.capture_authorized_payment",
                    "inputs": {"authorization_id": "AUTH-1", "body": {}},
                    "expects": ["capture_id"],
                },
                {
                    "step": 2,
                    "purpose": "refund the capture",
                    "tool_id": "payments.refund_captured_payment",
                    "inputs": {"capture_id": "{{capture_id}}", "body": {}},
                    "expects": ["refund_id"],
                },
            ],
        }
    )
    agent = await build_agent("Payments", registry, retriever, handler, llm)
    result = await agent.run(
        DomainAgentInput(user_request="capture AUTH-1 then refund it", domain="Payments")
    )

    assert result.status == "success"
    assert seen == [
        "/v2/payments/authorizations/AUTH-1/capture",
        "/v2/payments/captures/CAP-77/refund",
    ]
    assert result.context["refund_id"] == "REF-77"
    assert result.plan.is_multi_step
    await agent.aclose()


async def test_hallucinated_tool_ids_are_rejected(registry, retriever):
    llm = FakeLLM(
        {"steps": [{"step": 1, "tool_id": "payments.definitely_not_a_real_tool", "inputs": {}}]},
        {"steps": []},
    )
    agent = await build_agent(
        "Payments", registry, retriever, lambda r: httpx.Response(200, json={}), llm
    )
    result = await agent.run(DomainAgentInput(user_request="do something", domain="Payments"))

    assert result.status == "failed"
    assert result.next_action == "replan"
    assert not result.outcomes
    await agent.aclose()


async def test_planner_can_ask_for_clarification(registry, retriever):
    llm = FakeLLM({"clarification": "Which invoice should I send?", "steps": []})
    agent = await build_agent(
        "Invoices", registry, retriever, lambda r: httpx.Response(200, json={}), llm
    )
    result = await agent.run(DomainAgentInput(user_request="send it", domain="Invoices"))

    assert result.status == "needs_input"
    assert result.next_action == "ask_user"
    assert "invoice" in result.message.lower()
    await agent.aclose()


# --------------------------------------------------------------- execution --
async def test_api_failure_stops_the_plan_and_requests_recovery(registry, retriever):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"name": "UNPROCESSABLE_ENTITY", "message": "AUTHORIZATION_NOT_CAPTURED"}
        )

    llm = FakeLLM(
        {
            "steps": [
                {
                    "step": 1,
                    "tool_id": "payments.refund_captured_payment",
                    "inputs": {"capture_id": "CAP-1", "body": {}},
                },
                {
                    "step": 2,
                    "tool_id": "payments.show_refund_details",
                    "inputs": {"refund_id": "{{refund_id}}"},
                },
            ]
        }
    )
    agent = await build_agent("Payments", registry, retriever, handler, llm)
    result = await agent.run(DomainAgentInput(user_request="refund CAP-1", domain="Payments"))

    assert result.status == "recovery_required"
    assert result.next_action == "recover"
    assert len(result.outcomes) == 1  # step 2 never ran
    assert "AUTHORIZATION_NOT_CAPTURED" in result.errors[0]
    assert result.plan.steps[1].status == "pending"
    await agent.aclose()


async def test_unresolved_reference_fails_before_calling_the_api(registry, retriever):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    llm = FakeLLM(
        {
            "steps": [
                {
                    "step": 1,
                    "tool_id": "payments.show_captured_payment_details",
                    "inputs": {"capture_id": "{{capture_id}}"},
                }
            ]
        }
    )
    agent = await build_agent("Payments", registry, retriever, handler, llm)
    result = await agent.run(DomainAgentInput(user_request="show the capture", domain="Payments"))

    assert result.status == "recovery_required"
    assert "capture_id" in result.outcomes[0].summary
    assert calls == []
    await agent.aclose()


async def test_context_from_the_caller_is_usable(registry, retriever):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"id": "CAP-5", "status": "COMPLETED"})

    llm = FakeLLM(
        {
            "steps": [
                {
                    "step": 1,
                    "tool_id": "payments.show_captured_payment_details",
                    "inputs": {"capture_id": "{{capture_id}}"},
                }
            ]
        }
    )
    agent = await build_agent("Payments", registry, retriever, handler, llm)
    result = await agent.run(
        DomainAgentInput(
            user_request="show the capture", domain="Payments", context={"capture_id": "CAP-5"}
        )
    )

    assert result.status == "success"
    assert seen == ["/v2/payments/captures/CAP-5"]
    await agent.aclose()


# ---------------------------------------------------------------- scoping ---
async def test_agent_sees_a_shortlist_not_the_whole_registry(registry, retriever):
    llm = FakeLLM({"steps": []})
    agent = await build_agent(
        "Disputes", registry, retriever, lambda r: httpx.Response(200, json={}), llm
    )
    result = await agent.run(
        DomainAgentInput(user_request="show dispute details", domain="Disputes", top_k=4)
    )

    assert result.tools_discovered == len(registry.get_tools_by_domain("Disputes"))
    assert result.candidates_considered <= 4
    prompt = llm.prompts[0]
    assert prompt.count("tool_id:") <= 4
    assert "invoices." not in prompt  # other domains never reach the prompt
    await agent.aclose()


async def test_agent_works_without_an_llm(registry, retriever):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "CAP-3", "status": "COMPLETED"})

    agent = await build_agent("Payments", registry, retriever, handler, llm=None)
    agent._llm_resolved = True  # force the offline path regardless of environment
    result = await agent.run(
        DomainAgentInput(
            user_request="show captured payment details",
            domain="Payments",
            context={"capture_id": "CAP-3"},
        )
    )

    assert result.status == "success"
    assert result.plan.steps[0].tool_id == "payments.show_captured_payment_details"
    assert not result.llm_calls
    await agent.aclose()
