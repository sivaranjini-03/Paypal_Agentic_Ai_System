"""Steps 8 & 12 verification: the LangGraph workflow, recovery edges, cross-domain runs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.observability import Telemetry
from app.recovery.handler import RecoveryAction, RecoveryPolicy
from app.system import build_system
from app.tools.registry import ToolRegistry

BASE_URL = "https://api.example.test"


class ScriptedLLM:
    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append("\n".join(content for _, content in messages))
        payload = self.responses.pop(0) if self.responses else {}
        return SimpleNamespace(
            content=json.dumps(payload),
            usage_metadata={"input_tokens": 40, "output_tokens": 8},
        )


async def no_sleep(_seconds: float) -> None:
    """Recovery backoff without the wall-clock cost."""


def settings() -> Settings:
    return Settings(
        paypal_base_url=BASE_URL, paypal_client_id="cid", paypal_client_secret="shh"
    )


def make_workflow(handler, llm, *, policy: RecoveryPolicy | None = None, telemetry=None):
    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 600})
        return handler(request)

    return build_system(
        settings=settings(),
        registry=ToolRegistry.from_collection(),
        telemetry=telemetry or Telemetry(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(wrapped)),
        llm=llm,
        policy=policy,
        sleep=no_sleep,
    )


def route(*domains: tuple[str, str]) -> dict:
    return {"assignments": [{"domain": d, "objective": o} for d, o in domains]}


def steps(*entries: dict) -> dict:
    return {"steps": list(entries)}


# ---------------------------------------------------------------- happy path --
async def test_single_domain_workflow_runs_end_to_end():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [], "total_items": 0})

    llm = ScriptedLLM(
        route(("Disputes", "look for open disputes")),
        steps({"step": 1, "tool_id": "disputes.list_disputes", "inputs": {}}),
        {"answer": "You have no open disputes."},
    )
    workflow = make_workflow(handler, llm)
    state = await workflow.run("do I have open disputes?")

    assert state.workflow_status == "completed"
    assert state.tool_calls == ["disputes.list_disputes"]
    assert state.final_response == "You have no open disputes."
    assert state.total_tokens > 0
    await workflow.aclose()


async def test_multi_step_plan_advances_through_the_graph():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/capture"):
            return httpx.Response(201, json={"id": "CAP-1", "status": "COMPLETED"})
        return httpx.Response(201, json={"id": "REF-1", "status": "COMPLETED"})

    llm = ScriptedLLM(
        route(("Payments", "capture then refund")),
        steps(
            {
                "step": 1,
                "tool_id": "payments.capture_authorized_payment",
                "inputs": {"authorization_id": "AUTH-1", "body": {}},
            },
            {
                "step": 2,
                "tool_id": "payments.refund_captured_payment",
                "inputs": {"capture_id": "{{capture_id}}", "body": {}},
            },
        ),
        {"answer": "Captured and refunded."},
    )
    workflow = make_workflow(handler, llm)
    state = await workflow.run("capture AUTH-1 and refund it")

    assert state.workflow_status == "completed"
    assert state.tool_calls == [
        "payments.capture_authorized_payment",
        "payments.refund_captured_payment",
    ]
    assert state.context["capture_id"] == "CAP-1"
    assert state.context["refund_id"] == "REF-1"
    await workflow.aclose()


# --------------------------------------------------------------- cross-domain --
async def test_cross_domain_workflow_hands_identifiers_over():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if "/checkout/orders/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "id": "ORDER-9",
                    "purchase_units": [{"payments": {"captures": [{"id": "CAP-9"}]}}],
                },
            )
        return httpx.Response(201, json={"id": "REF-9", "status": "COMPLETED"})

    llm = ScriptedLLM(
        route(("Orders", "find the capture"), ("Payments", "refund it")),
        steps(
            {
                "step": 1,
                "tool_id": "orders.show_order_details",
                "inputs": {"order_id": "ORDER-9"},
                "expects": ["capture_id"],
            }
        ),
        steps(
            {
                "step": 1,
                "tool_id": "payments.refund_captured_payment",
                "inputs": {"capture_id": "{{capture_id}}", "body": {}},
            }
        ),
        {"answer": "Refunded CAP-9."},
    )
    workflow = make_workflow(handler, llm)
    state = await workflow.run("find the payment for order ORDER-9 and refund it")

    assert state.workflow_status == "completed"
    assert state.completed_domains == ["Orders", "Payments"]
    assert seen == ["/v2/checkout/orders/ORDER-9", "/v2/payments/captures/CAP-9/refund"]
    assert state.context["capture_id"] == "CAP-9"
    await workflow.aclose()


# ------------------------------------------------------------------ recovery --
async def test_transient_failure_is_retried_then_succeeds():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"message": "service unavailable"})
        return httpx.Response(200, json={"id": "CAP-2", "status": "COMPLETED"})

    llm = ScriptedLLM(
        route(("Payments", "show the capture")),
        steps(
            {
                "step": 1,
                "tool_id": "payments.show_captured_payment_details",
                "inputs": {"capture_id": "CAP-2"},
            }
        ),
        {"answer": "Capture CAP-2 is completed."},
    )
    workflow = make_workflow(handler, llm)
    state = await workflow.run("show capture CAP-2")

    assert state.workflow_status == "completed"
    assert attempts["count"] == 2
    assert state.retry_counts["transient"] == 1
    assert state.last_decision.action is RecoveryAction.WAIT_AND_RETRY
    await workflow.aclose()


async def test_expired_token_triggers_one_refresh_then_succeeds():
    calls = {"token": 0, "api": 0}

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": f"t{calls['token']}", "expires_in": 600})
        calls["api"] += 1
        if calls["api"] == 1:
            return httpx.Response(401, json={"error": "invalid_token"})
        return httpx.Response(200, json={"id": "CAP-3", "status": "COMPLETED"})

    llm = ScriptedLLM(
        route(("Payments", "show the capture")),
        steps(
            {
                "step": 1,
                "tool_id": "payments.show_captured_payment_details",
                "inputs": {"capture_id": "CAP-3"},
            }
        ),
        {"answer": "Capture CAP-3 is completed."},
    )
    workflow = build_system(
        settings=settings(),
        registry=ToolRegistry.from_collection(),
        telemetry=Telemetry(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(wrapped)),
        llm=llm,
        sleep=no_sleep,
    )
    state = await workflow.run("show capture CAP-3")

    assert state.workflow_status == "completed"
    assert calls["token"] == 2  # refreshed once
    assert state.retry_counts["auth"] == 1
    await workflow.aclose()


async def test_workflow_error_causes_a_replan_with_feedback():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/refund"):
            return httpx.Response(
                422,
                json={
                    "name": "UNPROCESSABLE_ENTITY",
                    "message": "cannot refund",
                    "details": [{"issue": "AUTHORIZATION_NOT_CAPTURED"}],
                },
            )
        return httpx.Response(201, json={"id": "CAP-4", "status": "COMPLETED"})

    llm = ScriptedLLM(
        route(("Payments", "refund the payment")),
        steps(
            {
                "step": 1,
                "tool_id": "payments.refund_captured_payment",
                "inputs": {"capture_id": "CAP-X", "body": {}},
            }
        ),
        # replan: capture first, then refund the new capture
        steps(
            {
                "step": 1,
                "tool_id": "payments.capture_authorized_payment",
                "inputs": {"authorization_id": "AUTH-4", "body": {}},
            }
        ),
        {"answer": "Captured the authorization first."},
    )
    workflow = make_workflow(handler, llm)
    state = await workflow.run("refund the payment for authorization AUTH-4")

    assert state.retry_counts["replan"] == 1
    assert state.workflow_status == "completed"
    assert state.tool_calls == [
        "payments.refund_captured_payment",
        "payments.capture_authorized_payment",
    ]
    replan_prompt = llm.prompts[2]
    assert "AUTHORIZATION_NOT_CAPTURED" in replan_prompt
    await workflow.aclose()


async def test_business_errors_are_not_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            422,
            json={
                "name": "UNPROCESSABLE_ENTITY",
                "message": "already refunded",
                "details": [{"issue": "CAPTURE_FULLY_REFUNDED"}],
            },
        )

    llm = ScriptedLLM(
        route(("Payments", "refund")),
        steps(
            {
                "step": 1,
                "tool_id": "payments.refund_captured_payment",
                "inputs": {"capture_id": "CAP-5", "body": {}},
            }
        ),
        {"answer": "That capture has already been fully refunded."},
    )
    workflow = make_workflow(handler, llm)
    state = await workflow.run("refund capture CAP-5")

    assert calls["count"] == 1  # no blind retry
    assert state.workflow_status == "failed"
    assert state.last_decision.action is RecoveryAction.FAIL
    assert "CAPTURE_FULLY_REFUNDED" in state.errors[-1]
    await workflow.aclose()


async def test_retries_are_bounded():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, json={"message": "down"})

    llm = ScriptedLLM(
        route(("Payments", "show the capture")),
        steps(
            {
                "step": 1,
                "tool_id": "payments.show_captured_payment_details",
                "inputs": {"capture_id": "CAP-6"},
            }
        ),
        {"answer": "The service is unavailable."},
    )
    workflow = make_workflow(handler, llm, policy=RecoveryPolicy(max_transient_retries=2))
    state = await workflow.run("show capture CAP-6")

    assert calls["count"] == 3  # initial + 2 retries
    assert state.workflow_status == "failed"
    await workflow.aclose()


async def test_missing_information_asks_the_user_instead_of_guessing():
    llm = ScriptedLLM(
        route(("Payments", "show the capture")),
        steps(
            {
                "step": 1,
                "tool_id": "payments.show_captured_payment_details",
                "inputs": {"capture_id": "{{capture_id}}"},
            }
        ),
    )
    workflow = make_workflow(lambda r: httpx.Response(200, json={}), llm)
    state = await workflow.run("show me that capture")

    assert state.workflow_status == "needs_input"
    assert "capture_id" in state.final_response
    await workflow.aclose()


async def test_unroutable_request_ends_cleanly():
    llm = ScriptedLLM({"assignments": [], "clarification": "I cannot help with the weather."})
    workflow = make_workflow(lambda r: httpx.Response(200, json={}), llm)
    state = await workflow.run("what is the weather in Madrid?")

    assert state.workflow_status == "needs_input"
    assert not state.tool_calls
    await workflow.aclose()


# --------------------------------------------------------------- telemetry ---
async def test_every_run_is_traceable():
    telemetry = Telemetry()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    llm = ScriptedLLM(
        route(("Disputes", "list disputes")),
        steps({"step": 1, "tool_id": "disputes.list_disputes", "inputs": {}}),
        {"answer": "No disputes."},
    )
    workflow = make_workflow(handler, llm, telemetry=telemetry)
    state = await workflow.run("list my disputes")

    events = telemetry.events(request_id=state.request_id, limit=50)
    kinds = {event.kind for event in events}
    assert {"workflow", "tool_call"} <= kinds
    assert any(event.tool_id == "disputes.list_disputes" for event in events)
    blob = json.dumps([e.model_dump(mode="json") for e in events]).lower()
    assert "secret" not in blob and "bearer" not in blob
    await workflow.aclose()


async def test_capability_tools_are_callable_through_the_workflow():
    llm = ScriptedLLM(
        route(("Knowledge", "explain the refund lifecycle")),
        steps(
            {
                "step": 1,
                "tool_id": "knowledge.search_documentation",
                "inputs": {"query": "refund requires capture", "top_k": 2},
            }
        ),
        {"answer": "Refunds apply to captures, so the payment must be captured first."},
    )
    workflow = make_workflow(lambda r: httpx.Response(500, json={}), llm)
    state = await workflow.run("can I refund a payment that was only authorized?")

    assert state.workflow_status == "completed"
    assert state.tool_calls == ["knowledge.search_documentation"]
    assert "capture" in state.outcomes[0].summary.lower()
    await workflow.aclose()
