"""Capabilities that are not HTTP APIs: knowledge retrieval and system search.

They are registered as ordinary tools so the agent discovers and selects them
through exactly the same path as the 116 collection APIs — no special casing in
any prompt.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.capabilities.rag import RagTool, rag_tool_definition
from app.capabilities.system_search import SystemSearchTool, system_tool_definition
from app.config import Settings, get_settings
from app.observability import Telemetry, get_telemetry
from app.tools.registry import ToolRegistry

InternalHandler = Callable[[dict[str, Any]], Awaitable[Any]]

__all__ = [
    "RagTool",
    "SystemSearchTool",
    "InternalHandler",
    "register_capabilities",
]


def register_capabilities(
    registry: ToolRegistry,
    *,
    telemetry: Telemetry | None = None,
    settings: Settings | None = None,
    embedder: Any | None = None,
) -> dict[str, InternalHandler]:
    """Add the non-API capabilities to a registry and return their handlers."""
    settings = settings or get_settings()
    telemetry = telemetry or get_telemetry()

    rag = RagTool(settings=settings, embedder=embedder)
    system = SystemSearchTool(registry=registry, telemetry=telemetry)

    definitions = [rag_tool_definition(), system_tool_definition()]
    handlers: dict[str, InternalHandler] = {
        definitions[0].id: rag.handle,
        definitions[1].id: system.handle,
    }

    for definition in definitions:
        if registry.get_tool(definition.id) is None:
            registry.register_tool(definition)
    return handlers
