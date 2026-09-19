"""MCP client.

Agents ask "what can this domain do?" and "run this tool" through here. No tool
name is ever hardcoded: everything comes from discovery.

Two transports behind one interface:
  * in-memory - the same FastMCP server object, no process/transport overhead
  * stdio     - a separately spawned server process, proving protocol compliance
"""

from __future__ import annotations

import sys
from typing import Any, Mapping

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import ClientTransport, StdioTransport
from mcp import types
from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT
from app.mcp.server import DomainToolServer, mcp_name
from app.tools.executor import ToolExecutor
from app.tools.models import ExecutionResult
from app.tools.registry import ToolRegistry


class DiscoveredTool(BaseModel):
    """A capability as advertised by an MCP server."""

    name: str
    tool_id: str
    title: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    domain: str = ""
    method: str = ""
    path: str = ""
    operation_type: str = ""
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)

    @classmethod
    def from_mcp(cls, tool: types.Tool) -> "DiscoveredTool":
        meta = tool.meta or {}
        return cls(
            name=tool.name,
            tool_id=meta.get("tool_id", tool.name.replace("-", ".")),
            title=tool.title or tool.name,
            description=tool.description or "",
            input_schema=tool.input_schema or {},
            domain=meta.get("domain", ""),
            method=meta.get("method", ""),
            path=meta.get("path", ""),
            operation_type=meta.get("operation_type", ""),
            requires=list(meta.get("requires", [])),
            produces=list(meta.get("produces", [])),
        )


def _to_execution_result(tool_id: str, result: CallToolResult) -> ExecutionResult:
    structured = result.structured_content
    if isinstance(structured, dict) and "tool_id" in structured:
        return ExecutionResult.model_validate(structured)
    text = ""
    for block in result.content or []:
        if isinstance(block, types.TextContent):
            text = block.text
            break
    return ExecutionResult(
        tool_id=tool_id,
        success=not result.is_error,
        error=text if result.is_error else None,
        error_type="mcp" if result.is_error else None,
        text=None if result.is_error else text,
    )


class MCPClient:
    """Discovery and invocation over a FastMCP transport."""

    def __init__(self, transport: ClientTransport | Any, *, domain: str | None = None) -> None:
        self.domain = domain
        self._client = Client(transport)
        self._server: DomainToolServer | None = None
        self._connected = False

    @classmethod
    def in_memory(
        cls,
        domain: str | None = None,
        *,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
    ) -> "MCPClient":
        server = DomainToolServer(registry, domain=domain, executor=executor)
        client = cls(server.mcp, domain=domain)
        client._server = server
        return client

    @classmethod
    def stdio(cls, domain: str | None = None, *, command: str | None = None) -> "MCPClient":
        args = ["-m", "app.mcp.server"]
        if domain:
            args += ["--domain", domain]
        transport = StdioTransport(
            command=command or sys.executable, args=args, cwd=str(PROJECT_ROOT)
        )
        return cls(transport, domain=domain)

    async def connect(self) -> "MCPClient":
        if not self._connected:
            await self._client.__aenter__()
            self._connected = True
        return self

    async def list_tools(self) -> list[DiscoveredTool]:
        return [DiscoveredTool.from_mcp(tool) for tool in await self._client.list_tools()]

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> ExecutionResult:
        result = await self._client.call_tool(
            mcp_name(name), dict(arguments or {}), raise_on_error=False
        )
        return _to_execution_result(name, result)

    async def refresh_credentials(self) -> None:
        """Drop cached credentials on the local server; remote servers manage their own."""
        if self._server is not None:
            self._server.executor.credentials.invalidate()

    async def close(self) -> None:
        if self._connected:
            await self._client.__aexit__(None, None, None)
            self._connected = False
        if self._server is not None:
            await self._server.aclose()

    async def __aenter__(self) -> "MCPClient":
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
