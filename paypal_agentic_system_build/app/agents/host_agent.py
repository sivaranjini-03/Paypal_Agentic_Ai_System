"""Host agent.

Owns the request end to end: decide which domains are involved, run their
agents in order, pass produced values between them, and turn the accumulated
results into an answer.

It never selects an API itself. Domain selection is its only routing decision;
tool selection belongs to the domain agents.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from app.agents.domain_agent import DomainAgent
from app.agents.models import AgentStatus, DomainAgentInput, DomainAgentResult
from app.config import Settings, get_settings
from app.llm import LLMCall, call_structured, get_chat_model, llm_available
from app.tools.registry import ToolRegistry, get_registry
from app.tools.retriever import ToolRetriever

MAX_DOMAINS = 3

ROUTER_SYSTEM = """You route a user request to the API domains that can serve it.

Rules:
- Choose only from the listed domains, at most {max_domains}.
- Order them by execution order: a domain that produces an identifier must come
  before the domain that consumes it.
- Give each domain a short objective describing its part of the request.
- Choose one domain when one suffices. Do not add domains "just in case".
- Route meta-questions about this agent's available tools, supported domains,
    capabilities, recent requests, activity, errors, or status to System. Do not
    route "what tools are available for managing <domain>" to that API domain.
- Route explanatory questions about API rules, payment lifecycle, eligibility,
    limits, errors, or how a PayPal operation works to Knowledge. Route a request
    to perform an operation to its API domain instead.
- If the request is not about any listed domain, return an empty list and explain
  in clarification.

Reply with JSON only:
{{"assignments": [{{"domain": "...", "objective": "..."}}],
  "reasoning": "...", "clarification": null}}"""

SYNTHESIS_SYSTEM = """You write the final reply to a user whose request was
executed against live APIs.

Rules:
- Use only the facts in the results. Never invent identifiers, amounts or states.
- Be concise and concrete: state what happened and the key identifiers.
- If something failed, say what failed and why, in plain language.

Reply with JSON only: {"answer": "..."}"""


class DomainAssignment(BaseModel):
    domain: str
    objective: str = ""


class RoutingDecision(BaseModel):
    assignments: list[DomainAssignment] = Field(default_factory=list)
    reasoning: str = ""
    clarification: str | None = None

    @property
    def domains(self) -> list[str]:
        return [assignment.domain for assignment in self.assignments]


class _Answer(BaseModel):
    answer: str = ""


class HostResult(BaseModel):
    """Everything the caller (chat UI, benchmark, API) needs about one request."""

    user_request: str
    status: AgentStatus
    detected_domains: list[str] = Field(default_factory=list)
    routing_reasoning: str = ""
    domain_results: list[DomainAgentResult] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    final_response: str = ""
    errors: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        own = sum(call.total_tokens for call in self.llm_calls)
        return own + sum(result.total_tokens for result in self.domain_results)

    @property
    def tool_calls(self) -> list[str]:
        return [o.tool_id for r in self.domain_results for o in r.outcomes]


class HostAgent:
    """Coordinates domain agents; holds the global view of one user request."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        retriever: ToolRetriever | None = None,
        llm: Any | None = None,
        settings: Settings | None = None,
        agent_factory: Callable[[str], DomainAgent] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        self.retriever = retriever or ToolRetriever(self.registry, settings=self.settings)
        self._llm = llm
        self._llm_resolved = llm is not None
        self._agent_factory = agent_factory or self._default_agent_factory
        self._agents: dict[str, DomainAgent] = {}

    def _default_agent_factory(self, domain: str) -> DomainAgent:
        return DomainAgent(domain, retriever=self.retriever, llm=self.llm, settings=self.settings)

    @property
    def llm(self) -> Any | None:
        if not self._llm_resolved:
            self._llm = get_chat_model(self.settings) if llm_available(self.settings) else None
            self._llm_resolved = True
        return self._llm

    def agent(self, domain: str) -> DomainAgent:
        if domain not in self._agents:
            self._agents[domain] = self._agent_factory(domain)
        return self._agents[domain]

    async def aclose(self) -> None:
        for agent in self._agents.values():
            await agent.aclose()
        self._agents.clear()

    # -------------------------------------------------------------- routing --
    def domain_catalogue(self) -> str:
        """The only tool-ish context the router sees: one line per domain."""
        return "\n".join(summary.as_prompt_line() for summary in self.registry.domain_summaries())

    def route(self, user_request: str, calls: list[LLMCall]) -> RoutingDecision:
        known = {domain.lower(): domain for domain in self.registry.list_domains()}

        if self.llm is not None:
            result = call_structured(
                self.llm,
                [
                    ("system", ROUTER_SYSTEM.format(max_domains=MAX_DOMAINS)),
                    (
                        "user",
                        f"Request: {user_request}\n\nDomains:\n{self.domain_catalogue()}",
                    ),
                ],
                RoutingDecision,
            )
            calls.append(result.call)
            if result.ok:
                decision: RoutingDecision = result.value
                decision.assignments = [
                    DomainAssignment(domain=known[a.domain.lower()], objective=a.objective)
                    for a in decision.assignments
                    if a.domain.lower() in known
                ][:MAX_DOMAINS]
                if decision.assignments or decision.clarification:
                    return decision

        domains = self.retriever.route_domains(user_request, top_n=1)
        return RoutingDecision(
            assignments=[DomainAssignment(domain=domain) for domain in domains],
            reasoning="lexical routing fallback",
            clarification=None if domains else "I could not match this request to a known domain.",
        )

    # ------------------------------------------------------------ synthesis --
    def _synthesize(
        self, user_request: str, result: HostResult, calls: list[LLMCall]
    ) -> str:
        facts = [
            {
                "domain": domain_result.domain,
                "status": domain_result.status,
                "steps": [
                    {"tool_id": o.tool_id, "success": o.success, "result": o.summary}
                    for o in domain_result.outcomes
                ],
                "message": domain_result.message,
            }
            for domain_result in result.domain_results
        ]
        if self.llm is None:
            return self._deterministic_answer(result)

        answer = call_structured(
            self.llm,
            [
                ("system", SYNTHESIS_SYSTEM),
                (
                    "user",
                    f"Request: {user_request}\n\nResults:\n{json.dumps(facts, default=str)[:6000]}",
                ),
            ],
            _Answer,
        )
        calls.append(answer.call)
        return answer.value.answer if answer.ok else self._deterministic_answer(result)

    @staticmethod
    def _deterministic_answer(result: HostResult) -> str:
        if result.status == "success":
            steps = ", ".join(result.tool_calls) or "no operations"
            return f"Completed via {steps}. Values: {json.dumps(result.context, default=str)[:400]}"
        if result.errors:
            return f"Could not complete the request: {result.errors[0]}"
        return "Could not complete the request."

    def synthesize_from_state(self, state: Any, succeeded: bool) -> tuple[str, list[LLMCall]]:
        """Answer a workflow run using only what the steps actually returned."""
        calls: list[LLMCall] = []
        facts = [
            {
                "tool_id": outcome.tool_id,
                "success": outcome.success,
                "result": outcome.summary,
            }
            for outcome in state.outcomes
        ]
        if self.llm is None or not facts:
            if succeeded and facts:
                return (
                    f"Completed via {', '.join(state.tool_calls)}. "
                    f"Values: {json.dumps(state.context, default=str)[:400]}",
                    calls,
                )
            reason = state.errors[-1] if state.errors else "no operation could be completed"
            return f"Could not complete the request: {reason}", calls

        answer = call_structured(
            self.llm,
            [
                ("system", SYNTHESIS_SYSTEM),
                (
                    "user",
                    f"Request: {state.user_request}\n"
                    f"Succeeded: {succeeded}\n"
                    f"Errors: {json.dumps(state.errors[-2:], default=str)}\n"
                    f"Results:\n{json.dumps(facts, default=str)[:6000]}",
                ),
            ],
            _Answer,
        )
        calls.append(answer.call)
        if answer.ok and answer.value.answer:
            return answer.value.answer, calls
        reason = state.errors[-1] if state.errors else "unknown error"
        return (
            f"Completed via {', '.join(state.tool_calls)}." if succeeded else
            f"Could not complete the request: {reason}"
        ), calls

    # ------------------------------------------------------------------ run --
    async def handle(
        self, user_request: str, *, context: dict[str, Any] | None = None, top_k: int = 8
    ) -> HostResult:
        started = time.perf_counter()
        calls: list[LLMCall] = []
        result = HostResult(user_request=user_request, status="failed", context=dict(context or {}))

        decision = self.route(user_request, calls)
        result.detected_domains = decision.domains
        result.routing_reasoning = decision.reasoning

        if not decision.domains:
            result.status = "needs_input"
            result.final_response = (
                decision.clarification or "I could not match this request to a known domain."
            )
            result.llm_calls = calls
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        for position, assignment in enumerate(decision.assignments):
            domain = assignment.domain
            agent = self.agent(domain)
            domain_result = await agent.run(
                DomainAgentInput(
                    user_request=user_request,
                    domain=domain,
                    objective=assignment.objective,
                    downstream_domains=decision.domains[position + 1 :],
                    context=dict(result.context),
                    top_k=top_k,
                )
            )
            result.domain_results.append(domain_result)
            # Values produced upstream become inputs for the next domain.
            result.context.update(domain_result.context)
            result.errors.extend(domain_result.errors)

            if domain_result.status != "success":
                result.status = domain_result.status
                break
        else:
            result.status = "success"

        if result.status in ("success", "partial"):
            result.final_response = self._synthesize(user_request, result, calls)
        elif result.status == "needs_input":
            result.final_response = next(
                (r.message for r in result.domain_results if r.status == "needs_input"),
                "More information is needed.",
            )
        else:
            result.final_response = self._deterministic_answer(result)

        result.llm_calls = calls
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
