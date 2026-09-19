"""Typed workflow state shared by every graph node."""

from app.state.state import AgentState, DomainAssignmentState, WorkflowStatus

__all__ = ["AgentState", "DomainAssignmentState", "WorkflowStatus"]
