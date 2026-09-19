"""Composition root.

Everything is wired here so no component has to know how the others are built:
registry -> capabilities -> executor -> MCP -> agents -> workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx

from app.agents.domain_agent import DomainAgent
from app.agents.host_agent import HostAgent
from app.capabilities import register_capabilities
from app.config import Settings, get_settings
from app.mcp.client import MCPClient
from app.observability import Telemetry, get_telemetry
from app.recovery.handler import RecoveryHandler, RecoveryPolicy
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.retriever import LLMReranker, ToolRetriever
from app.workflows.graph import AgentWorkflow


def build_system(
    *,
    settings: Settings | None = None,
    registry: ToolRegistry | None = None,
    telemetry: Telemetry | None = None,
    http_client: httpx.AsyncClient | None = None,
    llm: Any | None = None,
    rerank: bool = False,
    policy: RecoveryPolicy | None = None,
    sleep: Any | None = None,
) -> AgentWorkflow:
    settings = settings or get_settings()
    telemetry = get_telemetry() if telemetry is None else telemetry
    registry = ToolRegistry.from_collection() if registry is None else registry

    # Non-API capabilities join the registry before anything indexes it.
    handlers = register_capabilities(registry, telemetry=telemetry, settings=settings)

    executor = ToolExecutor(
        settings=settings,
        client=http_client,
        internal_handlers=handlers,
        telemetry=telemetry,
    )
    retriever = ToolRetriever(
        registry,
        settings=settings,
        reranker=LLMReranker(llm=llm, settings=settings) if rerank else None,
    )

    def agent_factory(domain: str) -> DomainAgent:
        return DomainAgent(
            domain,
            client=MCPClient.in_memory(domain, registry=registry, executor=executor),
            retriever=retriever,
            llm=llm,
            settings=settings,
        )

    host = HostAgent(
        registry=registry,
        retriever=retriever,
        llm=llm,
        settings=settings,
        agent_factory=agent_factory,
    )
    return AgentWorkflow(
        host,
        recovery=RecoveryHandler(policy),
        telemetry=telemetry,
        sleep=sleep,
    )


async def _run(request: str, *, verbose: bool) -> None:
    workflow = build_system()
    try:
        state = await workflow.run(request)
    finally:
        await workflow.aclose()

    print(f"\n{state.final_response}\n")
    print(f"status   : {state.workflow_status}")
    print(f"domains  : {', '.join(state.detected_domains) or '-'}")
    print(f"tools    : {', '.join(state.tool_calls) or '-'}")
    print(f"tokens   : {state.total_tokens}")
    if state.errors:
        print(f"errors   : {state.errors[-1]}")
    if verbose:
        print("context  :", json.dumps(state.context, default=str)[:600])
        for line in state.trace():
            print("  -", line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the agentic system")
    parser.add_argument("request", help="Natural language request")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args.request, verbose=args.verbose))


if __name__ == "__main__":
    main()
