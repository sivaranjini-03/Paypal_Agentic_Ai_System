"""Steps 10-11 verification: RAG tool and System Search tool."""

from __future__ import annotations

import json

import pytest

from app.capabilities import register_capabilities
from app.capabilities.rag import RAG_TOOL_ID, RagTool
from app.capabilities.system_search import SYSTEM_TOOL_ID, SystemSearchTool
from app.observability import Telemetry
from app.tools.registry import ToolRegistry
from app.tools.retriever import ToolRetriever
from app.tools.schema import build_input_schema


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.from_collection()


@pytest.fixture(scope="module")
def rag() -> RagTool:
    return RagTool(embedder=None)


# --------------------------------------------------------------------- RAG --
def test_knowledge_base_is_chunked_by_heading(rag):
    rag.load()
    assert len(rag._texts) > 10
    assert all(len(text) >= 80 for text in rag._texts)


def test_rag_finds_lifecycle_guidance(rag):
    result = rag.search("can I refund a payment that was only authorized?", top_k=3)
    assert result.passages
    combined = " ".join(p.text for p in result.passages).lower()
    assert "capture" in combined
    assert result.passages[0].source


def test_rag_finds_error_guidance(rag):
    result = rag.search("what does HTTP 429 mean and what should I do", top_k=3)
    combined = " ".join(p.text for p in result.passages).lower()
    assert "retry-after" in combined or "rate limited" in combined


async def test_rag_handler_returns_context_for_the_agent(rag):
    payload = await rag.handle({"query": "invoice must be sent before it can be paid", "top_k": 2})
    assert payload["passages"] and payload["context"]
    assert len(payload["passages"]) <= 2
    assert payload["documents_searched"] >= 1


async def test_rag_handler_validates_input(rag):
    assert (await rag.handle({"query": "  "}))["error"]


# ----------------------------------------------------------- system search --
def test_system_search_answers_capability_questions(registry):
    tool = SystemSearchTool(registry=registry, telemetry=Telemetry())
    result = tool.search("what tools are available for managing invoices?")

    assert result.scope == "capabilities"
    assert result.total_tools == len(registry)
    assert result.capabilities
    assert any(hit.domain == "Invoices" for hit in result.capabilities)


def test_system_search_answers_activity_questions(registry):
    telemetry = Telemetry()
    telemetry.log(
        "req-1",
        "tool_call",
        tool_id="payments.refund_captured_payment",
        status="error",
        latency_ms=120,
        error_type="http",
        error_message="CAPTURE_FULLY_REFUNDED",
    )
    tool = SystemSearchTool(registry=registry, telemetry=telemetry)
    result = tool.search("what was the status of my last request?")

    assert result.scope == "activity"
    assert result.activity and "refund_captured_payment" in result.activity[0]


def test_system_search_scope_can_be_forced(registry):
    tool = SystemSearchTool(registry=registry, telemetry=Telemetry())
    result = tool.search("disputes", scope="all")
    assert result.scope == "all"
    assert result.capabilities


# ------------------------------------------------------------ registration --
def test_capabilities_register_as_ordinary_discoverable_tools(registry):
    scratch = ToolRegistry.from_collection()
    before = len(scratch)
    handlers = register_capabilities(scratch, telemetry=Telemetry())

    assert len(scratch) == before + 2
    assert set(handlers) == {RAG_TOOL_ID, SYSTEM_TOOL_ID}
    assert "Knowledge" in scratch.list_domains()
    assert "System" in scratch.list_domains()

    schema = build_input_schema(scratch.require_tool(RAG_TOOL_ID))
    assert schema["properties"]["query"]["type"] == "string"
    assert "query" in schema["required"]


def test_capability_tools_are_retrievable_like_any_other_tool():
    scratch = ToolRegistry.from_collection()
    register_capabilities(scratch, telemetry=Telemetry())
    retriever = ToolRetriever(scratch, embedder=None)

    docs = retriever.retrieve("how does the refund lifecycle work", domains=["Knowledge"], top_k=2)
    assert docs and docs[0].tool_id == RAG_TOOL_ID

    system = retriever.retrieve("which tools exist for invoices", domains=["System"], top_k=2)
    assert system and system[0].tool_id == SYSTEM_TOOL_ID


def test_capabilities_do_not_require_credentials():
    scratch = ToolRegistry.from_collection()
    register_capabilities(scratch, telemetry=Telemetry())
    for tool_id in (RAG_TOOL_ID, SYSTEM_TOOL_ID):
        tool = scratch.require_tool(tool_id)
        assert tool.auth.required is False
        assert "secret" not in json.dumps(tool.model_dump(mode="json")).lower()
