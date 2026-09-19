"""System search tool.

Answers questions about the system itself: which capabilities exist, what a
domain can do, and what happened on recent requests. It reads the registry and
the telemetry log; it never calls an external API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.observability import Telemetry, get_telemetry
from app.tools.models import Parameter, ToolDefinition
from app.tools.registry import ToolRegistry, get_registry

SYSTEM_TOOL_ID = "system.search_system"
ACTIVITY_HINTS = (
    "last request",
    "recent",
    "status of",
    "what happened",
    "history",
    "log",
    "error",
    "failed",
    "activity",
)


class CapabilityHit(BaseModel):
    tool_id: str
    name: str
    domain: str
    operation: str
    call: str
    score: float = 0.0


class SystemSearchResult(BaseModel):
    query: str
    scope: str
    total_tools: int = 0
    domains: list[dict[str, Any]] = Field(default_factory=list)
    capabilities: list[CapabilityHit] = Field(default_factory=list)
    activity: list[str] = Field(default_factory=list)


def system_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        id=SYSTEM_TOOL_ID,
        name="Search system capabilities and activity",
        domain="System",
        hierarchy=["System"],
        description=(
            "Introspect this agent system. Use it to answer questions about what the "
            "system can do (which tools or domains exist for a topic, how many tools "
            "are available) and about its own activity (status of the last request, "
            "recent tool calls, recent errors). Returns system metadata, not customer data."
        ),
        summary="Query the tool registry and the telemetry log",
        method="INTERNAL",
        url_template="internal://system/search",
        path_template="/system/search",
        query_parameters=[
            Parameter(
                name="query",
                location="query",
                required=True,
                description="What to look up, e.g. 'tools for managing invoices' or 'my last request'",
            ),
            Parameter(
                name="scope",
                location="query",
                required=False,
                description="capabilities | activity | all (default: inferred from the query)",
            ),
            Parameter(
                name="limit",
                location="query",
                required=False,
                type="integer",
                description="Maximum results per section (default 8)",
            ),
        ],
        auth={"required": False, "type": "none", "scheme_source": "none"},
        operation_type="read",
        resource="system",
        source="internal",
    )


class SystemSearchTool:
    def __init__(
        self, *, registry: ToolRegistry | None = None, telemetry: Telemetry | None = None
    ) -> None:
        self.registry = registry or get_registry()
        self.telemetry = telemetry or get_telemetry()

    @staticmethod
    def infer_scope(query: str) -> str:
        lowered = query.lower()
        return "activity" if any(hint in lowered for hint in ACTIVITY_HINTS) else "capabilities"

    def search(self, query: str, scope: str | None = None, limit: int = 8) -> SystemSearchResult:
        scope = scope or self.infer_scope(query)
        result = SystemSearchResult(
            query=query, scope=scope, total_tools=len(self.registry)
        )

        if scope in ("capabilities", "all"):
            result.capabilities = [
                CapabilityHit(
                    tool_id=tool.id,
                    name=tool.name,
                    domain=tool.domain,
                    operation=tool.operation_type.value,
                    call=f"{tool.method} {tool.path_template}",
                    score=score,
                )
                for tool, score in self.registry.search_tools(query, limit=limit)
            ]
            matched = {hit.domain for hit in result.capabilities}
            result.domains = [
                summary.model_dump()
                for summary in self.registry.domain_summaries()
                if not matched or summary.domain in matched
            ]

        if scope in ("activity", "all"):
            request_ids = self.telemetry.requests(limit=1)
            events = self.telemetry.events(
                request_id=request_ids[0] if request_ids and "last" in query.lower() else None,
                limit=limit,
            )
            result.activity = [event.summary() for event in events]

        return result

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}
        scope = arguments.get("scope")
        limit = int(arguments.get("limit") or 8)
        return self.search(query, scope=str(scope) if scope else None, limit=limit).model_dump()
