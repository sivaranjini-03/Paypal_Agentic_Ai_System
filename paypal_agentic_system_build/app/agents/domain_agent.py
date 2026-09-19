"""Reusable domain agent.

One implementation serves every domain; a domain is configuration, not code.
The agent understands the request, asks MCP what exists, narrows that to a
handful of candidates, plans, executes through MCP and inspects the results.

It deliberately does not: hardcode tool names, build HTTP requests, hold
credentials, or decide cross-domain orchestration.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from app.agents.models import (
    DomainAgentInput,
    DomainAgentResult,
    Plan,
    PlanStep,
    PreparedPlan,
    StepOutcome,
)
from app.config import Settings, get_settings
from app.llm import LLMCall, call_structured, get_chat_model, llm_available
from app.mcp.client import DiscoveredTool, MCPClient
from app.tools.executor import extract_named, harvest_identifiers
from app.tools.retriever import ToolCandidate, ToolRetriever

REFERENCE_PATTERN = re.compile(r"^\{\{([^{}]+)\}\}$|^\$([A-Za-z_][A-Za-z0-9_]*)$")
MAX_RESULT_SUMMARY = 600

PLANNER_SYSTEM = """You plan API calls for the "{domain}" domain.

Rules:
- Use ONLY tool_ids from the candidate list. Never invent one.
- You handle only this domain's part of the request. Other domains handle theirs,
  so do not refuse because a later action is missing from your candidates.
- Answering a question IS calling the tool that returns the data. If a search or
  list tool can find the resource, call it instead of asking the user for an id.
- Prefer the fewest steps that satisfy your part of the request.
- If a tool needs a value you do not have, add an earlier step that produces it
  and reference it as "{{{{key}}}}" in the later step's inputs.
- Values already in known_context can be referenced the same way.
- Use "expects" to name the values a later step needs from this step's response.
- Only set "clarification" when no candidate tool can make progress at all.
- Request bodies go under the "body" input key.

Reply with JSON only:
{{"reasoning": "...", "clarification": null,
  "steps": [{{"step": 1, "purpose": "...", "tool_id": "...",
              "inputs": {{}}, "expects": ["..."]}}]}}"""


class _PlanResponse(BaseModel):
    reasoning: str = ""
    clarification: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)


def _summarize(value: Any, limit: int = MAX_RESULT_SUMMARY) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


class DomainAgent:
    """Plans and executes workflows inside a single domain."""

    def __init__(
        self,
        domain: str,
        *,
        client: MCPClient | None = None,
        retriever: ToolRetriever | None = None,
        llm: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.domain = domain
        self.settings = settings or get_settings()
        self.retriever = retriever or ToolRetriever(settings=self.settings)
        self._client = client
        self._owns_client = client is None
        self._llm = llm
        self._llm_resolved = llm is not None

    # ------------------------------------------------------------ lifecycle --
    async def client(self) -> MCPClient:
        if self._client is None:
            self._client = MCPClient.in_memory(self.domain, registry=self.retriever.registry)
        return await self._client.connect()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def llm(self) -> Any | None:
        if not self._llm_resolved:
            self._llm = get_chat_model(self.settings) if llm_available(self.settings) else None
            self._llm_resolved = True
        return self._llm

    # ------------------------------------------------------------- planning --
    def _candidates(
        self, request: str, context: Mapping[str, Any], top_k: int
    ) -> list[ToolCandidate]:
        """Retrieval is scoped to this domain; context is a hint, not a filter.

        Hard-filtering on available context would hide the very tools a
        multi-step plan needs in order to obtain the missing values.
        """
        return self.retriever.retrieve(request, domains=[self.domain], top_k=top_k)

    def _plan_with_llm(
        self,
        request: DomainAgentInput,
        candidates: Sequence[ToolCandidate],
        allowed: set[str],
        calls: list[LLMCall],
    ) -> Plan | None:
        listing = "\n\n".join(candidate.as_prompt_block() for candidate in candidates)
        known = json.dumps(
            {k: _summarize(v, 120) for k, v in request.context.items()}, default=str
        )
        downstream = (
            f"\nLater domains will handle: {', '.join(request.downstream_domains)}"
            if request.downstream_domains
            else ""
        )
        correction = (
            f"\n\nThe previous attempt failed. Fix it: {request.feedback}"
            if request.feedback
            else ""
        )
        result = call_structured(
            self.llm,
            [
                ("system", PLANNER_SYSTEM.format(domain=self.domain)),
                (
                    "user",
                    f"Overall request: {request.user_request}\n"
                    f"Your part: {request.objective or request.user_request}{downstream}\n\n"
                    f"known_context: {known}\n\nCandidates:\n{listing}{correction}",
                ),
            ],
            _PlanResponse,
        )
        calls.append(result.call)
        if not result.ok:
            return None

        response: _PlanResponse = result.value
        steps: list[PlanStep] = []
        for index, raw in enumerate(response.steps, start=1):
            tool_id = str(raw.get("tool_id", "")).strip()
            if tool_id not in allowed:
                continue  # hallucinated or out-of-domain tool
            steps.append(
                PlanStep(
                    step=int(raw.get("step", index)),
                    purpose=str(raw.get("purpose", "")),
                    tool_id=tool_id,
                    inputs=dict(raw.get("inputs") or {}),
                    expects=[str(k) for k in (raw.get("expects") or [])],
                )
            )
        return Plan(
            steps=steps, reasoning=response.reasoning, clarification=response.clarification
        )

    def _plan_without_llm(
        self, candidates: Sequence[ToolCandidate], context: Mapping[str, Any]
    ) -> Plan:
        """Deterministic fallback: best candidate whose inputs the context satisfies."""
        for candidate in candidates:
            if all(key in context for key in candidate.requires):
                inputs = {
                    name: context[name]
                    for name in candidate.required_parameters
                    if name in context
                }
                return Plan(
                    steps=[
                        PlanStep(
                            step=1,
                            purpose=f"Best lexical/semantic match for the request",
                            tool_id=candidate.tool_id,
                            inputs=inputs,
                            expects=candidate.produces,
                        )
                    ],
                    reasoning="planned without an LLM: highest ranked satisfiable candidate",
                )
        return Plan(clarification="No available tool can run with the information provided.")

    # ------------------------------------------------------------ execution --
    @staticmethod
    def _resolve(value: Any, context: Mapping[str, Any]) -> Any:
        if isinstance(value, str):
            match = REFERENCE_PATTERN.match(value.strip())
            if match:
                key = match.group(1) or match.group(2)
                return context.get(key, value)
            return re.sub(
                r"\{\{([^{}]+)\}\}",
                lambda m: str(context.get(m.group(1), m.group(0))),
                value,
            )
        if isinstance(value, dict):
            return {k: DomainAgent._resolve(v, context) for k, v in value.items()}
        if isinstance(value, list):
            return [DomainAgent._resolve(v, context) for v in value]
        return value

    def _unresolved(self, inputs: Mapping[str, Any]) -> list[str]:
        missing: list[str] = []
        blob = json.dumps(inputs, default=str)
        for match in re.finditer(r"\{\{([^{}]+)\}\}", blob):
            missing.append(match.group(1))
        return sorted(set(missing))

    async def _execute_step(
        self, step: PlanStep, context: dict[str, Any], client: MCPClient
    ) -> StepOutcome:
        inputs = self._resolve(step.inputs, context)
        unresolved = self._unresolved(inputs)
        if unresolved:
            return StepOutcome(
                step=step.step,
                tool_id=step.tool_id,
                success=False,
                summary=f"missing values: {', '.join(unresolved)}",
            )

        result = await client.call_tool(step.tool_id, inputs)
        produced = dict(result.produced_context)
        if result.success:
            # Declared metadata wins; the plan's `expects` and harvested ids fill gaps.
            for source in (
                extract_named(result.data, step.expects) if step.expects else {},
                harvest_identifiers(result.data),
            ):
                for key, value in source.items():
                    produced.setdefault(key, value)
        return StepOutcome(
            step=step.step,
            tool_id=step.tool_id,
            success=result.success,
            summary=result.brief(MAX_RESULT_SUMMARY),
            produced_context=produced,
            result=result,
        )

    # ------------------------------------------------------------------ run --
    async def prepare(self, request: DomainAgentInput) -> PreparedPlan:
        """Discover, retrieve and plan. No side effects on the outside world."""
        client = await self.client()
        discovered: list[DiscoveredTool] = await client.list_tools()
        allowed = {tool.tool_id for tool in discovered}
        candidates = self._candidates(request.user_request, request.context, request.top_k)

        calls: list[LLMCall] = []
        plan: Plan | None = None
        if self.llm is not None:
            plan = self._plan_with_llm(request, candidates, allowed, calls)
        if plan is None:
            plan = self._plan_without_llm(candidates, request.context)

        return PreparedPlan(
            plan=plan,
            candidates=[candidate.tool_id for candidate in candidates],
            tools_discovered=len(discovered),
            llm_calls=calls,
        )

    async def execute_step(self, step: PlanStep, context: dict[str, Any]) -> StepOutcome:
        return await self._execute_step(step, context, await self.client())

    async def refresh_credentials(self) -> None:
        client = await self.client()
        await client.refresh_credentials()

    async def run(self, request: DomainAgentInput) -> DomainAgentResult:
        prepared = await self.prepare(request)
        plan = prepared.plan

        outcome = DomainAgentResult(
            domain=self.domain,
            status="failed",
            plan=plan,
            context=dict(request.context),
            candidates_considered=len(prepared.candidates),
            tools_discovered=prepared.tools_discovered,
            llm_calls=prepared.llm_calls,
        )

        if plan.clarification:
            outcome.status = "needs_input"
            outcome.next_action = "ask_user"
            outcome.message = plan.clarification
            return outcome

        if not plan.steps:
            outcome.status = "failed"
            outcome.message = "no executable plan could be produced for this domain"
            outcome.next_action = "replan"
            return outcome

        for step in plan.steps[: request.max_steps]:
            step.status = "running"
            step_outcome = await self.execute_step(step, outcome.context)
            outcome.outcomes.append(step_outcome)

            if not step_outcome.success:
                step.status = "failed"
                outcome.status = "recovery_required"
                outcome.next_action = "recover"
                outcome.errors.append(
                    f"step {step.step} ({step.tool_id}): {step_outcome.summary}"
                )
                outcome.message = step_outcome.summary
                return outcome

            step.status = "succeeded"
            outcome.context.update(step_outcome.produced_context)

        outcome.status = "success"
        outcome.next_action = "none"
        outcome.message = outcome.outcomes[-1].summary if outcome.outcomes else ""
        return outcome
