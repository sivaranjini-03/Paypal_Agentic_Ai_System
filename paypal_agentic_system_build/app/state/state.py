"""Workflow state.

One typed object holds everything the workflow needs to run, resume, debug and
recover. Nothing important lives in Python objects between nodes: if it matters
to the workflow, it is here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.models import Plan, PlanStep, StepOutcome
from app.llm import LLMCall
from app.observability import new_request_id
from app.recovery.handler import RecoveryDecision

WorkflowStatus = Literal[
    "started",
    "routing",
    "planning",
    "executing",
    "recovering",
    "needs_input",
    "completed",
    "failed",
]


class DomainAssignmentState(BaseModel):
    domain: str
    objective: str = ""


class AgentState(BaseModel):
    """Shared state for the whole request."""

    request_id: str = Field(default_factory=new_request_id)
    user_request: str = ""

    # routing
    detected_domains: list[str] = Field(default_factory=list)
    assignments: list[DomainAssignmentState] = Field(default_factory=list)
    domain_index: int = 0
    routing_reasoning: str = ""

    # planning / execution
    current_domain: str = ""
    current_objective: str = ""
    retrieved_tools: list[str] = Field(default_factory=list)
    tools_discovered: int = 0
    plan: Plan = Field(default_factory=Plan)
    current_step: int = 0
    context: dict[str, Any] = Field(default_factory=dict)
    outcomes: list[StepOutcome] = Field(default_factory=list)
    completed_domains: list[str] = Field(default_factory=list)

    # recovery
    retry_counts: dict[str, int] = Field(default_factory=dict)
    last_decision: RecoveryDecision | None = None
    replan_feedback: str = ""
    errors: list[str] = Field(default_factory=list)

    # outcome
    workflow_status: WorkflowStatus = "started"
    final_response: str = ""
    clarification: str = ""
    llm_calls: list[LLMCall] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.llm_calls)

    @property
    def tool_calls(self) -> list[str]:
        return [outcome.tool_id for outcome in self.outcomes]

    @property
    def current_assignment(self) -> DomainAssignmentState | None:
        if 0 <= self.domain_index < len(self.assignments):
            return self.assignments[self.domain_index]
        return None

    @property
    def pending_step(self) -> PlanStep | None:
        steps = self.plan.steps
        return steps[self.current_step] if 0 <= self.current_step < len(steps) else None

    @property
    def downstream_domains(self) -> list[str]:
        return [a.domain for a in self.assignments[self.domain_index + 1 :]]

    def bump(self, key: str) -> dict[str, int]:
        counts = dict(self.retry_counts)
        counts[key] = counts.get(key, 0) + 1
        return counts

    def trace(self) -> list[str]:
        """Human-readable step trace, useful in tests and the CLI."""
        return [
            f"{outcome.tool_id}: {'ok' if outcome.success else 'failed'} - {outcome.summary[:120]}"
            for outcome in self.outcomes
        ]
