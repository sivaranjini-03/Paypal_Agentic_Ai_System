"""The LangGraph workflow.

    route -> select domain -> plan -> execute -> evaluate -> [next | recover] -> respond

Control flow lives in the graph and data lives in `AgentState`, so a run can be
inspected, checkpointed and resumed, and recovery is an explicit edge rather
than a try/except buried in an agent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.domain_agent import DomainAgent
from app.agents.host_agent import HostAgent
from app.agents.models import DomainAgentInput, Plan
from app.observability import Telemetry, get_telemetry
from app.recovery.handler import RecoveryAction, RecoveryHandler, classify
from app.state.state import AgentState, DomainAssignmentState

NODE_ROUTE = "route"
NODE_SELECT = "select_domain"
NODE_PLAN = "plan"
NODE_EXECUTE = "execute"
NODE_EVALUATE = "evaluate"
NODE_RECOVER = "recover"
NODE_RESPOND = "respond"


class AgentWorkflow:
    """Owns the compiled graph and the agents it drives."""

    def __init__(
        self,
        host: HostAgent,
        *,
        recovery: RecoveryHandler | None = None,
        telemetry: Telemetry | None = None,
        sleep: Callable[[float], Any] | None = None,
        max_steps: int = 8,
    ) -> None:
        self.host = host
        self.recovery = recovery or RecoveryHandler()
        self.telemetry = telemetry or get_telemetry()
        self._sleep = sleep or asyncio.sleep
        self.max_steps = max_steps
        self.graph = self._build().compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------- plumbing --
    def _agent(self, state: AgentState) -> DomainAgent:
        agent = self.host.agent(state.current_domain)
        return agent

    def _bind_request(self, state: AgentState, agent: DomainAgent) -> None:
        """Tag executor telemetry with the request id."""
        client = agent._client
        server = getattr(client, "_server", None)
        if server is not None:
            server.executor.request_id = state.request_id

    async def aclose(self) -> None:
        await self.host.aclose()

    # ---------------------------------------------------------------- nodes --
    async def route(self, state: AgentState) -> dict[str, Any]:
        calls = list(state.llm_calls)
        decision = self.host.route(state.user_request, calls)
        self.telemetry.log(
            state.request_id,
            "workflow",
            name=NODE_ROUTE,
            status="ok" if decision.domains else "unroutable",
            detail={"domains": decision.domains},
        )
        return {
            "assignments": [
                DomainAssignmentState(domain=a.domain, objective=a.objective)
                for a in decision.assignments
            ],
            "detected_domains": decision.domains,
            "routing_reasoning": decision.reasoning,
            "clarification": decision.clarification or "",
            "llm_calls": calls,
            "workflow_status": "routing" if decision.domains else "needs_input",
        }

    async def select_domain(self, state: AgentState) -> dict[str, Any]:
        assignment = state.current_assignment
        if assignment is None:
            return {"workflow_status": "completed"}
        return {
            "current_domain": assignment.domain,
            "current_objective": assignment.objective,
            "workflow_status": "planning",
            "current_step": 0,
        }

    async def plan(self, state: AgentState) -> dict[str, Any]:
        agent = self._agent(state)
        prepared = await agent.prepare(
            DomainAgentInput(
                user_request=state.user_request,
                domain=state.current_domain,
                objective=state.current_objective,
                downstream_domains=state.downstream_domains,
                context=dict(state.context),
                feedback=state.replan_feedback,
            )
        )
        self._bind_request(state, agent)
        self.telemetry.log(
            state.request_id,
            "workflow",
            name=NODE_PLAN,
            domain=state.current_domain,
            status="planned" if prepared.plan.steps else "no_plan",
            detail={
                "steps": [step.tool_id for step in prepared.plan.steps],
                "candidates": prepared.candidates,
            },
        )
        return {
            "plan": prepared.plan,
            "retrieved_tools": prepared.candidates,
            "tools_discovered": prepared.tools_discovered,
            "llm_calls": [*state.llm_calls, *prepared.llm_calls],
            "current_step": 0,
            "replan_feedback": "",
            "clarification": prepared.plan.clarification or "",
            "workflow_status": "executing" if prepared.plan.steps else "needs_input",
        }

    async def execute(self, state: AgentState) -> dict[str, Any]:
        step = state.pending_step
        if step is None:
            return {"workflow_status": "executing"}

        agent = self._agent(state)
        self._bind_request(state, agent)
        step.status = "running"
        outcome = await agent.execute_step(step, dict(state.context))
        step.status = "succeeded" if outcome.success else "failed"

        context = dict(state.context)
        context.update(outcome.produced_context)
        return {
            "outcomes": [*state.outcomes, outcome],
            "context": context,
            "plan": state.plan,
            "workflow_status": "executing",
        }

    async def evaluate(self, state: AgentState) -> dict[str, Any]:
        outcome = state.outcomes[-1] if state.outcomes else None
        if outcome is None or not outcome.success:
            return {"workflow_status": "recovering"}

        next_step = state.current_step + 1
        if next_step < len(state.plan.steps) and next_step < self.max_steps:
            return {"current_step": next_step, "workflow_status": "executing"}

        # Domain finished: drop its plan so the graph moves on instead of re-running it.
        return {
            "domain_index": state.domain_index + 1,
            "completed_domains": [*state.completed_domains, state.current_domain],
            "plan": Plan(),
            "current_step": 0,
            "workflow_status": "executing",
        }

    async def recover(self, state: AgentState) -> dict[str, Any]:
        outcome = state.outcomes[-1] if state.outcomes else None
        step = state.pending_step

        if outcome is None or outcome.result is None:
            message = outcome.summary if outcome else "the step could not be executed"
            return {
                "workflow_status": "needs_input",
                "clarification": f"I need more information: {message}",
                "errors": [*state.errors, message],
            }

        error = classify(outcome.result)
        decision = self.recovery.decide(error, state.retry_counts)
        feedback = RecoveryHandler.feedback(error, decision)

        self.telemetry.log(
            state.request_id,
            "recovery",
            name=decision.action.value,
            domain=state.current_domain,
            tool_id=error.tool_id,
            status=error.category.value,
            error_type=error.category.value,
            error_message=error.message[:300],
            retry_count=state.retry_counts.get(RecoveryHandler.counter_key(decision.action), 0),
            detail={"reason": decision.reason, "issues": error.issue_codes[:3]},
        )

        update: dict[str, Any] = {
            "last_decision": decision,
            "retry_counts": state.bump(RecoveryHandler.counter_key(decision.action)),
            "errors": [*state.errors, error.describe()],
            "workflow_status": "recovering",
        }

        if decision.action is RecoveryAction.WAIT_AND_RETRY:
            await self._sleep(decision.delay_seconds)
        if decision.action is RecoveryAction.REFRESH_AUTH:
            await self._agent(state).refresh_credentials()
        if decision.action in (RecoveryAction.REPLAN, RecoveryAction.FIX_PARAMETERS):
            update["replan_feedback"] = feedback
        if decision.action is RecoveryAction.ASK_USER:
            update["clarification"] = (
                f"{decision.reason}. Missing: {', '.join(decision.missing_parameters) or 'details'}"
            )
            update["workflow_status"] = "needs_input"
        if decision.action is RecoveryAction.FAIL:
            update["workflow_status"] = "failed"

        if step is not None and decision.action in (
            RecoveryAction.RETRY,
            RecoveryAction.WAIT_AND_RETRY,
            RecoveryAction.REFRESH_AUTH,
        ):
            step.status = "pending"
            update["plan"] = state.plan
        return update

    async def respond(self, state: AgentState) -> dict[str, Any]:
        if state.workflow_status == "needs_input":
            return {
                "final_response": state.clarification or "I need more information to continue.",
                "workflow_status": "needs_input",
            }

        # Success is "every assigned domain finished and the last call worked",
        # not "nothing ever failed" - a recovered failure is still a success.
        completed_all = bool(state.assignments) and len(state.completed_domains) >= len(
            state.assignments
        )
        succeeded = completed_all and bool(state.outcomes) and state.outcomes[-1].success
        answer, calls = self.host.synthesize_from_state(state, succeeded)
        self.telemetry.log(
            state.request_id,
            "workflow",
            name=NODE_RESPOND,
            status="completed" if succeeded else "failed",
            detail={"tools": state.tool_calls},
        )
        return {
            "final_response": answer,
            "llm_calls": [*state.llm_calls, *calls],
            "workflow_status": "completed" if succeeded else "failed",
        }

    # ----------------------------------------------------------- transitions --
    def after_route(self, state: AgentState) -> str:
        return NODE_SELECT if state.assignments else NODE_RESPOND

    def after_select(self, state: AgentState) -> str:
        return NODE_PLAN if state.current_assignment is not None else NODE_RESPOND

    def after_plan(self, state: AgentState) -> str:
        return NODE_EXECUTE if state.plan.steps else NODE_RESPOND

    def after_evaluate(self, state: AgentState) -> str:
        if state.workflow_status == "recovering":
            return NODE_RECOVER
        if state.pending_step is not None:
            return NODE_EXECUTE
        return NODE_SELECT if state.current_assignment is not None else NODE_RESPOND

    def after_recover(self, state: AgentState) -> str:
        decision = state.last_decision
        if decision is None or decision.terminal:
            return NODE_RESPOND
        if decision.action in (RecoveryAction.REPLAN, RecoveryAction.FIX_PARAMETERS):
            return NODE_PLAN
        return NODE_EXECUTE

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node(NODE_ROUTE, self.route)
        graph.add_node(NODE_SELECT, self.select_domain)
        graph.add_node(NODE_PLAN, self.plan)
        graph.add_node(NODE_EXECUTE, self.execute)
        graph.add_node(NODE_EVALUATE, self.evaluate)
        graph.add_node(NODE_RECOVER, self.recover)
        graph.add_node(NODE_RESPOND, self.respond)

        graph.add_edge(START, NODE_ROUTE)
        graph.add_conditional_edges(
            NODE_ROUTE, self.after_route, {NODE_SELECT: NODE_SELECT, NODE_RESPOND: NODE_RESPOND}
        )
        graph.add_conditional_edges(
            NODE_SELECT, self.after_select, {NODE_PLAN: NODE_PLAN, NODE_RESPOND: NODE_RESPOND}
        )
        graph.add_conditional_edges(
            NODE_PLAN, self.after_plan, {NODE_EXECUTE: NODE_EXECUTE, NODE_RESPOND: NODE_RESPOND}
        )
        graph.add_edge(NODE_EXECUTE, NODE_EVALUATE)
        graph.add_conditional_edges(
            NODE_EVALUATE,
            self.after_evaluate,
            {
                NODE_EXECUTE: NODE_EXECUTE,
                NODE_SELECT: NODE_SELECT,
                NODE_RECOVER: NODE_RECOVER,
                NODE_RESPOND: NODE_RESPOND,
            },
        )
        graph.add_conditional_edges(
            NODE_RECOVER,
            self.after_recover,
            {
                NODE_EXECUTE: NODE_EXECUTE,
                NODE_PLAN: NODE_PLAN,
                NODE_RESPOND: NODE_RESPOND,
            },
        )
        graph.add_edge(NODE_RESPOND, END)
        return graph

    # ------------------------------------------------------------------ run --
    async def run(
        self, user_request: str, *, context: dict[str, Any] | None = None, thread_id: str | None = None
    ) -> AgentState:
        state = AgentState(user_request=user_request, context=dict(context or {}))
        config = {"configurable": {"thread_id": thread_id or state.request_id}, "recursion_limit": 60}
        final = await self.graph.ainvoke(state, config=config)
        return AgentState.model_validate(final)
