"""Structured contracts exchanged between agents.

Plans are data, not prose: every step names a discovered tool and its inputs, so
the workflow engine can execute, resume, inspect and replan them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm import LLMCall
from app.tools.models import ExecutionResult

StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
AgentStatus = Literal["success", "partial", "needs_input", "recovery_required", "failed"]

# Inputs may reference values produced by earlier steps, e.g. "{{capture_id}}".
CONTEXT_REFERENCE = "{{%s}}"


class PlanStep(BaseModel):
    step: int
    purpose: str = ""
    tool_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    expects: list[str] = Field(
        default_factory=list, description="Context keys this step is expected to produce"
    )
    status: StepStatus = "pending"


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    reasoning: str = ""
    clarification: str | None = Field(
        default=None, description="Set when the request cannot be planned without the user"
    )

    @property
    def is_multi_step(self) -> bool:
        return len(self.steps) > 1


class StepOutcome(BaseModel):
    step: int
    tool_id: str
    success: bool
    summary: str = ""
    produced_context: dict[str, Any] = Field(default_factory=dict)
    result: ExecutionResult | None = None


class DomainAgentInput(BaseModel):
    user_request: str
    domain: str
    objective: str = Field(
        default="", description="This domain's part of a possibly larger workflow"
    )
    downstream_domains: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    feedback: str = Field(
        default="", description="Why the previous attempt failed, for replanning"
    )
    top_k: int = 8
    max_steps: int = 6


class PreparedPlan(BaseModel):
    """Planning output, separated from execution so a graph can drive the steps."""

    plan: Plan = Field(default_factory=Plan)
    candidates: list[str] = Field(default_factory=list)
    tools_discovered: int = 0
    llm_calls: list[LLMCall] = Field(default_factory=list)


class DomainAgentResult(BaseModel):
    domain: str
    status: AgentStatus
    plan: Plan = Field(default_factory=Plan)
    outcomes: list[StepOutcome] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    candidates_considered: int = 0
    tools_discovered: int = 0
    errors: list[str] = Field(default_factory=list)
    message: str = ""
    next_action: Literal["replan", "ask_user", "recover", "none"] = "none"
    llm_calls: list[LLMCall] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.llm_calls)

    @property
    def failed_outcome(self) -> StepOutcome | None:
        return next((o for o in self.outcomes if not o.success), None)
