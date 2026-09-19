"""JSON Schema generation for tool inputs.

One place converts a `ToolDefinition` into the input schema used by MCP tool
discovery and by LLM parameter generation, so both always agree.
"""

from __future__ import annotations

from typing import Any

from app.tools.models import Parameter, ToolDefinition

BODY_ARGUMENT = "body"


def _property(parameter: Parameter) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": parameter.type}
    description = parameter.description or f"{parameter.location} parameter"
    if parameter.example:
        description = f"{description} (example: {parameter.example})"
    schema["description"] = description[:500]
    return schema


def build_input_schema(tool: ToolDefinition) -> dict[str, Any]:
    """JSON Schema describing every runtime input the executor accepts.

    Credentials are excluded by construction: they are injected downstream.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in tool.path_parameters:
        properties[parameter.name] = _property(parameter)
        required.append(parameter.name)

    for parameter in tool.query_parameters:
        properties.setdefault(parameter.name, _property(parameter))
        if parameter.required:
            required.append(parameter.name)

    for parameter in tool.headers:
        if not parameter.required:
            continue  # optional headers are noise in a prompt
        properties.setdefault(parameter.name, _property(parameter))
        required.append(parameter.name)

    if tool.request_body_schema:
        body_schema = dict(tool.request_body_schema)
        body_schema.setdefault("description", "Request payload")
        properties[BODY_ARGUMENT] = body_schema
        if tool.method in ("POST", "PUT", "PATCH") and body_schema.get("type") == "object":
            required.append(BODY_ARGUMENT)

    return {
        "type": "object",
        "properties": properties,
        "required": sorted(set(required)),
        "additionalProperties": False,
    }
