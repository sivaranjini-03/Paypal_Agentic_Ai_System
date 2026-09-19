"""Step 1 verification: Postman parsing, hierarchy preservation and secret safety."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.models import OperationType
from app.tools.postman_parser import parse_collection

COLLECTION = Path(__file__).resolve().parents[1] / "data" / "PayPal APIs.postman_collection.json"


@pytest.fixture(scope="module")
def result():
    return parse_collection(COLLECTION)


@pytest.fixture()
def nested_collection(tmp_path: Path) -> Path:
    collection = {
        "info": {"name": "Synthetic", "schema": "v2.1.0"},
        "auth": {"type": "bearer", "bearer": [{"key": "token", "value": "{{access_token}}"}]},
        "item": [
            {
                "name": "Level1",
                "item": [
                    {
                        "name": "Level2",
                        "item": [
                            {
                                "name": "Level3",
                                "item": [
                                    {
                                        "name": "Deep request",
                                        "event": [
                                            {
                                                "listen": "test",
                                                "script": {
                                                    "exec": [
                                                        'pm.collectionVariables.set("thing_id", x);'
                                                    ]
                                                },
                                            }
                                        ],
                                        "request": {
                                            "method": "POST",
                                            "header": [
                                                {"key": "Authorization", "value": "Bearer abc123"},
                                                {
                                                    "key": "Content-Type",
                                                    "value": "application/json",
                                                },
                                            ],
                                            "body": {
                                                "mode": "raw",
                                                "raw": '{"amount": 10, "currency": "USD"}',
                                            },
                                            "url": {
                                                "raw": "{{base_url}}/v2/things/:thing_id/activate?dry_run=true",
                                                "host": ["{{base_url}}"],
                                                "path": ["v2", "things", ":thing_id", "activate"],
                                                "variable": [
                                                    {
                                                        "key": "thing_id",
                                                        "value": "{{thing_id}}",
                                                        "description": "(Required) The thing id.",
                                                    }
                                                ],
                                                "query": [
                                                    {"key": "dry_run", "value": "true"},
                                                ],
                                            },
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    {"name": "Broken request", "request": {"method": "GET"}},
                ],
            }
        ],
    }
    path = tmp_path / "synthetic.postman_collection.json"
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# real collection
# --------------------------------------------------------------------------- #
def test_parses_every_request_in_the_collection(result):
    assert result.parsed_count == 116
    assert result.skipped_count == 0


def test_tool_ids_are_unique_and_deterministic(result):
    ids = [t.id for t in result.tools]
    assert len(ids) == len(set(ids))
    again = parse_collection(COLLECTION)
    assert [t.id for t in again.tools] == ids
    assert [t.fingerprint for t in again.tools] == [t.fingerprint for t in result.tools]


def test_domains_come_from_the_postman_hierarchy(result):
    domains = {t.domain for t in result.tools}
    assert {"Payments", "Invoices", "Disputes", "Orders", "Subscriptions"} <= domains
    nested = [t for t in result.tools if len(t.hierarchy) > 1]
    assert nested, "nested folders must be preserved"
    for tool in result.tools:
        assert tool.domain == tool.hierarchy[0]


def test_every_tool_is_structurally_valid(result):
    for tool in result.tools:
        assert tool.id and tool.name and tool.method and tool.domain
        assert tool.path_template.startswith("/")
        for param in tool.parameters:
            assert param.name
            assert param.location in {"path", "query", "header", "body"}


def test_path_parameters_are_templated(result):
    tool = next(t for t in result.tools if t.name == "Refund captured payment")
    assert tool.method == "POST"
    assert tool.path_template == "/v2/payments/captures/{capture_id}/refund"
    assert [p.name for p in tool.path_parameters] == ["capture_id"]
    assert tool.operation_type is OperationType.ACTION


def test_query_parameters_and_operation_type(result):
    tool = next(t for t in result.tools if t.name == "List transactions")
    assert tool.operation_type is OperationType.LIST
    names = {p.name for p in tool.query_parameters}
    assert {"start_date", "end_date"} <= names


def test_request_body_schema_is_inferred(result):
    tool = next(t for t in result.tools if t.name == "Create order")
    assert tool.body_mode == "raw"
    assert tool.request_body_schema is not None
    assert "intent" in tool.request_body_schema["properties"]


def test_workflow_dependency_metadata(result):
    create_order = next(t for t in result.tools if t.name == "Create order")
    assert "order_id" in create_order.produces

    refund = next(t for t in result.tools if t.name == "Refund captured payment")
    assert "capture_id" in refund.requires


def test_no_credentials_leak_into_the_registry(result):
    """Credential *values* from the collection must never reach the registry."""
    raw = json.loads(COLLECTION.read_text(encoding="utf-8"))
    secret_values = {
        v["value"]
        for v in raw.get("variable", [])
        if v.get("value")
        and isinstance(v["value"], str)
        and len(v["value"]) > 12
        and any(h in v["key"].lower() for h in ("secret", "token", "client_id", "password"))
    }
    blob = json.dumps(result.model_dump(mode="json"))
    for secret in secret_values:
        assert secret not in blob, "credential material leaked into the registry"
    assert "client_secret" not in json.dumps(result.variables)
    for tool in result.tools:
        assert all(p.name.lower() != "authorization" for p in tool.headers)
        assert all(
            p.example is None or "{{" not in (p.example or "") or not p.variable_ref
            for p in tool.parameters
            if p.variable_ref and "secret" in p.variable_ref
        )


def test_auth_metadata_is_inherited_and_overridden(result):
    tool = next(t for t in result.tools if t.name == "Generate access_token")
    assert tool.auth.type == "basic"
    assert tool.auth.scheme_source == "request"
    other = next(t for t in result.tools if t.name == "Show order details")
    assert other.auth.required and other.auth.type == "bearer"


def test_semantic_text_is_retrieval_ready(result):
    tool = next(t for t in result.tools if t.name == "Refund captured payment")
    text = tool.semantic_text()
    assert "Domain: Payments" in text
    assert "capture_id" in text
    assert "POST /v2/payments/captures/{capture_id}/refund" in text


# --------------------------------------------------------------------------- #
# synthetic collection
# --------------------------------------------------------------------------- #
def test_deeply_nested_folders_are_traversed(nested_collection):
    result = parse_collection(nested_collection)
    tool = next(t for t in result.tools if t.name == "Deep request")
    assert tool.hierarchy == ["Level1", "Level2", "Level3"]
    assert tool.domain == "Level1"
    assert tool.path_template == "/v2/things/{thing_id}/activate"
    assert tool.produces == ["thing_id"]
    assert tool.request_body_schema["required"] == ["amount", "currency"]
    assert all(h.name.lower() != "authorization" for h in tool.headers)
    assert "abc123" not in json.dumps(result.model_dump(mode="json"))


def test_malformed_requests_are_reported_not_fatal(nested_collection):
    result = parse_collection(nested_collection)
    assert result.parsed_count == 1
    assert result.skipped_count == 1
    assert any(i.item == "Broken request" for i in result.issues)


def test_new_apis_appear_without_code_changes(nested_collection, tmp_path: Path):
    """Adding a request to the collection must yield a new tool automatically."""
    before = parse_collection(nested_collection)
    collection = json.loads(nested_collection.read_text(encoding="utf-8"))
    collection["item"].append(
        {
            "name": "BrandNewDomain",
            "item": [
                {
                    "name": "Brand new API",
                    "request": {
                        "method": "GET",
                        "url": {"raw": "{{base_url}}/v9/new", "path": ["v9", "new"]},
                    },
                }
            ],
        }
    )
    path = tmp_path / "extended.postman_collection.json"
    path.write_text(json.dumps(collection), encoding="utf-8")

    after = parse_collection(path)
    assert after.parsed_count == before.parsed_count + 1
    assert any(t.domain == "BrandNewDomain" for t in after.tools)
