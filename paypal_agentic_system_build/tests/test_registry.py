"""Step 2 verification: registry build, persistence, structural filtering and search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.models import OperationType, ToolDefinition
from app.tools.registry import RegistryDocument, ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "data" / "PayPal APIs.postman_collection.json"


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.from_collection(COLLECTION)


def test_registry_is_generated_from_the_collection(registry):
    assert len(registry) == 116
    assert registry.document.collection_name == "PayPal APIs"
    assert sum(registry.stats().values()) == len(registry)


def test_domains_and_categories(registry):
    domains = registry.list_domains()
    assert "Payments" in domains and "Invoices" in domains
    assert registry.get_tools_by_domain("payments") == registry.get_tools_by_domain("Payments")
    assert any("Invoices > Templates" == c for c in registry.list_categories("Invoices"))


def test_lookup_by_id(registry):
    tool = registry.require_tool("payments.refund_captured_payment")
    assert tool.method == "POST"
    assert registry.get_tool("does.not.exist") is None
    with pytest.raises(KeyError):
        registry.require_tool("does.not.exist")


def test_dependency_indexes_support_planning(registry):
    producers = registry.tools_producing("capture_id")
    assert any(t.name == "Capture payment for order" for t in producers)
    consumers = registry.tools_requiring("capture_id")
    assert any(t.name == "Refund captured payment" for t in consumers)


def test_structural_filtering_narrows_the_candidate_set(registry):
    payments = registry.filter_tools(domains=["Payments"])
    assert 0 < len(payments) < len(registry)

    reads = registry.filter_tools(domains=["Payments"], operation_types=[OperationType.READ])
    assert reads and all(t.method == "GET" for t in reads)

    posts = registry.filter_tools(methods=["post"])
    assert posts and all(t.method == "POST" for t in posts)


def test_available_context_filter_excludes_unsatisfiable_tools(registry):
    runnable = registry.filter_tools(domains=["Payments"], available_context=["capture_id"])
    ids = {t.id for t in runnable}
    assert "payments.refund_captured_payment" in ids
    # Needs authorization_id, which the workflow does not have yet.
    assert "payments.void_authorized_payment" not in ids


def test_search_ranks_relevant_capabilities_first(registry):
    results = registry.search_tools("refund a captured payment", limit=5)
    assert results
    assert results[0][0].name == "Refund captured payment"
    assert results[0][1] > 0

    scoped = registry.search_tools("dispute", domains=["Disputes"], limit=3)
    assert scoped and all(t.domain == "Disputes" for t, _ in scoped)


def test_domain_summaries_are_compact_and_credential_free(registry):
    summaries = registry.domain_summaries()
    assert len(summaries) == len(registry.list_domains())
    blob = json.dumps([s.model_dump() for s in summaries]).lower()
    assert "secret" not in blob and "bearer" not in blob
    payments = next(s for s in summaries if s.domain == "Payments")
    assert payments.tool_count > 0 and payments.resources


def test_save_and_load_roundtrip_is_deterministic(registry, tmp_path: Path):
    path = registry.save(tmp_path / "tool_registry.json")
    reloaded = ToolRegistry.load(path)
    assert [t.id for t in reloaded.list_tools()] == [t.id for t in registry.list_tools()]
    assert path.read_text(encoding="utf-8") == registry.save(tmp_path / "again.json").read_text(
        encoding="utf-8"
    )


def test_non_collection_capabilities_can_be_registered(registry):
    scratch = ToolRegistry(RegistryDocument.model_validate(registry.document.model_dump()))
    scratch.register_tool(
        ToolDefinition(
            id="system.rag_search",
            name="Knowledge base search",
            domain="System",
            hierarchy=["System"],
            method="INTERNAL",
            url_template="internal://rag_search",
            path_template="/rag_search",
            source="internal",
        )
    )
    assert scratch.get_tool("system.rag_search") is not None
    assert "System" in scratch.list_domains()
    assert len(scratch) == len(registry) + 1
