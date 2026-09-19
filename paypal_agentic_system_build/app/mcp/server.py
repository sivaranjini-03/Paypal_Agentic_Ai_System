"""Dynamic MCP server (FastMCP).

Tools are generated from the registry at startup: there is no per-API handler
in this file. FastMCP's `@mcp.tool` decorator assumes tools are known at import
time, which is exactly what this system must avoid, so each capability is a
`Tool` subclass built from a `ToolDefinition` and dispatched to the generic
executor.

A server can be scoped to a domain, which is what turns "one giant tool server"
into the per-domain MCP servers the architecture calls for.

Responsibilities: expose tools, invoke tools. Planning happens above this layer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Mapping

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.base import ToolResult
from mcp import types
from pydantic import PrivateAttr

from app.tools.executor import ToolExecutor
from app.tools.models import ToolDefinition
from app.tools.registry import ToolRegistry, get_registry
from app.tools.schema import build_input_schema

SERVER_VERSION = "0.1.0"
INSTRUCTIONS = (
    "Tools are generated from an API collection and may change between sessions. "
    "Always discover them with list_tools; authentication is handled server-side."
)


def mcp_name(tool_id: str) -> str:
    """MCP tool names are restricted to [A-Za-z0-9_-]."""
    return tool_id.replace(".", "-")


def tool_description(tool: ToolDefinition, max_chars: int = 600) -> str:
    detail = tool.description or tool.name
    return f"[{tool.domain}] {tool.method} {tool.path_template}. {detail}"[:max_chars]


def tool_meta(tool: ToolDefinition) -> dict[str, Any]:
    """Discovery metadata: semantics and workflow dependencies, never credentials."""
    return {
        "tool_id": tool.id,
        "domain": tool.domain,
        "hierarchy": tool.hierarchy,
        "method": tool.method,
        "path": tool.path_template,
        "operation_type": tool.operation_type.value,
        "resource": tool.resource,
        "requires": tool.requires,
        "produces": tool.produces,
    }


class ApiTool(Tool):
    """A registry capability exposed as an MCP tool, executed generically."""

    _definition: ToolDefinition = PrivateAttr()
    _executor: ToolExecutor = PrivateAttr()

    @classmethod
    def build(cls, definition: ToolDefinition, executor: ToolExecutor) -> "ApiTool":
        tool = cls(
            name=mcp_name(definition.id),
            title=definition.name,
            description=tool_description(definition),
            parameters=build_input_schema(definition),
            tags={definition.domain, definition.operation_type.value},
            meta=tool_meta(definition),
        )
        tool._definition = definition
        tool._executor = executor
        return tool

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        result = await self._executor.execute(self._definition, arguments or {})
        payload = result.model_dump(mode="json")
        return ToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, default=str))],
            structured_content=payload,
            is_error=not result.success,
        )


class DomainToolServer:
    """A FastMCP server whose tool set is derived from the registry."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        domain: str | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.domain = domain
        self.executor = executor or ToolExecutor()

        definitions = (
            self.registry.get_tools_by_domain(domain) if domain else self.registry.list_tools()
        )
        if domain and not definitions:
            raise ValueError(f"no tools found for domain: {domain}")

        self._tools: dict[str, ApiTool] = {}
        for definition in definitions:
            tool = ApiTool.build(definition, self.executor)
            self._tools[tool.name] = tool

        self.mcp: FastMCP = FastMCP(
            name=self.name,
            version=SERVER_VERSION,
            instructions=INSTRUCTIONS,
            tools=list(self._tools.values()),
        )

    @property
    def name(self) -> str:
        return f"tools-{self.domain.lower().replace(' ', '-')}" if self.domain else "tools-all"

    def resolve(self, name: str) -> ToolDefinition | None:
        tool = self._tools.get(name) or self._tools.get(mcp_name(name))
        return tool.definition if tool else None

    def list_tool_descriptors(self) -> list[types.Tool]:
        return [tool.to_mcp_tool() for tool in self._tools.values()]

    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None) -> ToolResult:
        tool = self._tools.get(name) or self._tools.get(mcp_name(name))
        if tool is None:
            payload = {
                "success": False,
                "error": f"unknown tool: {name}",
                "error_type": "unknown_tool",
            }
            return ToolResult(
                content=[types.TextContent(type="text", text=json.dumps(payload))],
                structured_content=payload,
                is_error=True,
            )
        return await tool.run(dict(arguments or {}))

    async def aclose(self) -> None:
        await self.executor.aclose()


async def run_stdio(domain: str | None = None) -> None:
    server = DomainToolServer(domain=domain)
    try:
        await server.mcp.run_stdio_async(show_banner=False)
    finally:
        await server.aclose()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a dynamic MCP tool server over stdio")
    parser.add_argument("--domain", default=None, help="Expose only this domain's tools")
    args = parser.parse_args()
    asyncio.run(run_stdio(args.domain))


if __name__ == "__main__":
    _main()
