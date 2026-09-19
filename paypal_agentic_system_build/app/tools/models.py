"""Normalized, source-agnostic representation of an API capability.

These models are the single internal contract shared by every layer of the
system (registry, retriever, MCP server, executor, planner). Nothing here is
PayPal-specific: any source that can produce a `ToolDefinition` plugs in.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

ParameterLocation = Literal["path", "query", "header", "body"]


class OperationType(str, Enum):
    """Coarse capability class used for cheap metadata-level filtering."""

    LIST = "list"
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ACTION = "action"


class Parameter(BaseModel):
    """A single input accepted by a tool, in a specific location."""

    name: str
    location: ParameterLocation
    required: bool = False
    description: str = ""
    type: str = "string"
    example: str | None = None
    enabled: bool = True
    variable_ref: str | None = Field(
        default=None,
        description="Collection variable the source template bound to this parameter, if any.",
    )


class ResponseMetadata(BaseModel):
    """Lightweight response descriptor. Example bodies are deliberately dropped."""

    name: str = ""
    status: str = ""
    code: int | None = None


class AuthRequirement(BaseModel):
    """Auth metadata only. Credentials never live in a tool definition."""

    required: bool = True
    type: str = "bearer"
    scheme_source: Literal["collection", "folder", "request", "none"] = "collection"


class ToolDefinition(BaseModel):
    """Normalized capability derived from an API source definition."""

    id: str
    name: str

    domain: str
    hierarchy: list[str] = Field(default_factory=list)

    description: str = ""
    summary: str = ""

    method: str
    url_template: str
    path_template: str

    path_parameters: list[Parameter] = Field(default_factory=list)
    query_parameters: list[Parameter] = Field(default_factory=list)
    headers: list[Parameter] = Field(default_factory=list)

    body_mode: Literal["none", "raw", "urlencoded", "formdata"] = "none"
    body_content_type: str | None = None
    request_body_schema: dict[str, Any] | None = None

    response_metadata: list[ResponseMetadata] = Field(default_factory=list)

    auth: AuthRequirement = Field(default_factory=AuthRequirement)
    operation_type: OperationType = OperationType.ACTION
    resource: str = ""

    # Workflow dependency metadata (section 20 of the design contract).
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)

    source: str = ""
    source_ref: str | None = None
    fingerprint: str = ""

    @property
    def parameters(self) -> list[Parameter]:
        """All declared parameters across every location."""
        return [*self.path_parameters, *self.query_parameters, *self.headers]

    @property
    def required_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters if p.required]

    def semantic_text(self, max_description_chars: int = 900) -> str:
        """Retrieval-facing text. Contains semantics only, never credentials."""
        required = ", ".join(p.name for p in self.required_parameters) or "none"
        optional = ", ".join(p.name for p in self.parameters if not p.required) or "none"
        body_fields = ", ".join((self.request_body_schema or {}).get("properties", {})) or "none"
        description = (self.description or self.summary)[:max_description_chars]
        return "\n".join(
            [
                f"Tool: {self.name}",
                f"Domain: {self.domain}",
                f"Category: {' > '.join(self.hierarchy) or self.domain}",
                f"Operation: {self.operation_type.value} {self.resource}".strip(),
                f"HTTP: {self.method} {self.path_template}",
                f"Required parameters: {required}",
                f"Optional parameters: {optional}",
                f"Body fields: {body_fields}",
                f"Requires context: {', '.join(self.requires) or 'none'}",
                f"Produces context: {', '.join(self.produces) or 'none'}",
                f"Description: {description}",
            ]
        )


class ParserIssue(BaseModel):
    """A non-fatal problem encountered while parsing a source definition."""
    level: Literal["warning", "skipped"]
    item: str
    hierarchy: list[str] = Field(default_factory=list)
    reason: str


class ParseResult(BaseModel):
    """Outcome of parsing one source collection."""

    source: str
    collection_name: str = ""
    tools: list[ToolDefinition] = Field(default_factory=list)
    issues: list[ParserIssue] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)

    @property
    def parsed_count(self) -> int:
        return len(self.tools)

    @property
    def skipped_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "skipped")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    def report(self) -> str:
        return (
            f"Parsed: {self.parsed_count}\n"
            f"Skipped: {self.skipped_count}\n"
            f"Warnings: {self.warning_count}"
        )


class ExecutionResult(BaseModel):
    """Outcome of a single tool invocation, as seen by the agent and recovery layer."""

    tool_id: str
    success: bool
    status_code: int | None = None
    data: Any = None
    text: str | None = None
    error: str | None = None
    error_type: str | None = None
    latency_ms: float = 0.0
    attempt: int = 1
    request_summary: str = ""
    response_headers: dict[str, str] = Field(default_factory=dict)
    produced_context: dict[str, Any] = Field(default_factory=dict)

    def brief(self, max_chars: int = 1200) -> str:
        """Compact, prompt-safe rendering of the result."""
        import json as _json

        if self.success:
            payload = _json.dumps(self.data, default=str) if self.data is not None else ""
            return f"OK {self.status_code} {payload[:max_chars]}"
        return f"ERROR {self.status_code or ''} [{self.error_type}] {(self.error or '')[:max_chars]}"
