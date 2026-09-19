"""Failure classification and bounded recovery.

Classification and strategy are deterministic: the LLM is not asked whether a
504 is retryable. The model is only re-engaged for replanning, where judgement
is actually required.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.tools.models import ExecutionResult

FIELD_PATTERN = re.compile(r"/([a-z_][a-z0-9_]*)(?:/\d+)?$", re.IGNORECASE)
MISSING_HINTS = ("MISSING", "REQUIRED", "NOT_SUPPLIED", "CANNOT_BE_NULL")
ALREADY_HINTS = ("ALREADY", "DUPLICATE", "FULLY_REFUNDED", "COMPLETED_", "_COMPLETED")
PRECONDITION_HINTS = ("NOT_CAPTURED", "NOT_APPROVED", "NOT_ELIGIBLE", "INVALID_STATE", "NOT_PAID")


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    BUSINESS = "business"
    WORKFLOW = "workflow"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    WAIT_AND_RETRY = "wait_and_retry"
    REFRESH_AUTH = "refresh_auth"
    FIX_PARAMETERS = "fix_parameters"
    REPLAN = "replan"
    ASK_USER = "ask_user"
    FAIL = "fail"


class RecoveryPolicy(BaseModel):
    """Bounds. Configuration, not constants sprinkled through the code."""

    max_transient_retries: int = 3
    max_rate_limit_waits: int = 2
    max_auth_refresh: int = 1
    max_parameter_repairs: int = 1
    max_replans: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0

    def backoff(self, attempt: int) -> float:
        return min(self.base_delay_seconds * (2 ** max(0, attempt - 1)), self.max_delay_seconds)


class ErrorInfo(BaseModel):
    category: ErrorCategory
    status_code: int | None = None
    message: str = ""
    issue_codes: list[str] = Field(default_factory=list)
    retryable: bool = False
    retry_after: float | None = None
    missing_parameters: list[str] = Field(default_factory=list)
    tool_id: str = ""

    def describe(self) -> str:
        parts = [f"{self.category.value}"]
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        if self.issue_codes:
            parts.append(", ".join(self.issue_codes[:3]))
        return f"{' | '.join(parts)}: {self.message[:300]}"


class RecoveryDecision(BaseModel):
    action: RecoveryAction
    reason: str = ""
    delay_seconds: float = 0.0
    missing_parameters: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None

    @property
    def terminal(self) -> bool:
        return self.action in (RecoveryAction.FAIL, RecoveryAction.ASK_USER)


def _issue_codes(payload: Any) -> list[str]:
    codes: list[str] = []
    if isinstance(payload, dict):
        for detail in payload.get("details") or []:
            if isinstance(detail, dict) and detail.get("issue"):
                codes.append(str(detail["issue"]))
        if not codes and isinstance(payload.get("name"), str):
            codes.append(payload["name"])
        if isinstance(payload.get("error"), str):
            codes.append(payload["error"])
    return codes


def _fields(payload: Any, message: str) -> list[str]:
    fields: list[str] = []
    if isinstance(payload, dict):
        for detail in payload.get("details") or []:
            if not isinstance(detail, dict):
                continue
            pointer = detail.get("field") or detail.get("location") or ""
            match = FIELD_PATTERN.search(str(pointer))
            if match:
                fields.append(match.group(1))
    for match in re.finditer(r"parameter\(s\):\s*([a-z0-9_,\s]+)", message, re.IGNORECASE):
        fields.extend(part.strip() for part in match.group(1).split(",") if part.strip())
    return sorted(set(fields))


def classify(result: ExecutionResult) -> ErrorInfo:
    """Normalize any failure into one category with actionable detail."""
    message = result.error or ""
    codes = _issue_codes(result.data)
    haystack = f"{' '.join(codes)} {message}".upper()
    status = result.status_code
    retry_after = None
    if raw := result.response_headers.get("retry-after"):
        try:
            retry_after = float(raw)
        except ValueError:
            retry_after = None

    def info(category: ErrorCategory, **kwargs: Any) -> ErrorInfo:
        return ErrorInfo(
            category=category,
            status_code=status,
            message=message,
            issue_codes=codes,
            retry_after=retry_after,
            tool_id=result.tool_id,
            **kwargs,
        )

    if result.error_type in ("timeout", "network"):
        return info(ErrorCategory.TRANSIENT, retryable=True)
    if result.error_type == "validation" and status is None:
        return info(
            ErrorCategory.VALIDATION, missing_parameters=_fields(result.data, message)
        )
    if result.error_type == "authentication":
        return info(ErrorCategory.AUTHENTICATION, retryable=True)
    if result.error_type == "internal":
        return info(ErrorCategory.UNKNOWN)

    if status is None:
        return info(ErrorCategory.UNKNOWN)
    if status == 401:
        return info(ErrorCategory.AUTHENTICATION, retryable=True)
    if status == 403:
        return info(ErrorCategory.PERMISSION)
    if status == 404:
        return info(ErrorCategory.NOT_FOUND)
    if status == 429:
        return info(ErrorCategory.RATE_LIMIT, retryable=True)
    if status >= 500:
        return info(ErrorCategory.TRANSIENT, retryable=True)

    if status in (400, 409, 422):
        if any(hint in haystack for hint in ALREADY_HINTS):
            return info(ErrorCategory.BUSINESS)
        if any(hint in haystack for hint in PRECONDITION_HINTS):
            return info(ErrorCategory.WORKFLOW)
        missing = _fields(result.data, message)
        if missing or any(hint in haystack for hint in MISSING_HINTS):
            return info(ErrorCategory.VALIDATION, missing_parameters=missing)
        return info(ErrorCategory.BUSINESS)

    return info(ErrorCategory.UNKNOWN)


class RecoveryHandler:
    """Maps a classified error plus attempt counters onto one bounded action."""

    def __init__(self, policy: RecoveryPolicy | None = None) -> None:
        self.policy = policy or RecoveryPolicy()

    def decide(self, error: ErrorInfo, attempts: dict[str, int]) -> RecoveryDecision:
        policy = self.policy

        def used(key: str) -> int:
            return int(attempts.get(key, 0))

        if error.category is ErrorCategory.TRANSIENT:
            if used("transient") < policy.max_transient_retries:
                return RecoveryDecision(
                    action=RecoveryAction.WAIT_AND_RETRY,
                    delay_seconds=policy.backoff(used("transient") + 1),
                    reason="transient failure; retrying with exponential backoff",
                    error=error,
                )
            return RecoveryDecision(
                action=RecoveryAction.FAIL, reason="transient retries exhausted", error=error
            )

        if error.category is ErrorCategory.RATE_LIMIT:
            if used("rate_limit") < policy.max_rate_limit_waits:
                delay = error.retry_after or policy.backoff(used("rate_limit") + 1)
                return RecoveryDecision(
                    action=RecoveryAction.WAIT_AND_RETRY,
                    delay_seconds=min(delay, policy.max_delay_seconds),
                    reason="rate limited; honouring backoff before retrying",
                    error=error,
                )
            return RecoveryDecision(
                action=RecoveryAction.FAIL, reason="still rate limited after waiting", error=error
            )

        if error.category is ErrorCategory.AUTHENTICATION:
            if used("auth") < policy.max_auth_refresh:
                return RecoveryDecision(
                    action=RecoveryAction.REFRESH_AUTH,
                    reason="credentials rejected; refreshing the token once",
                    error=error,
                )
            return RecoveryDecision(
                action=RecoveryAction.FAIL,
                reason="authentication still failing after refresh",
                error=error,
            )

        if error.category is ErrorCategory.PERMISSION:
            return RecoveryDecision(
                action=RecoveryAction.FAIL,
                reason="the configured credentials lack permission for this operation",
                error=error,
            )

        if error.category is ErrorCategory.VALIDATION:
            if used("parameters") < policy.max_parameter_repairs:
                return RecoveryDecision(
                    action=RecoveryAction.FIX_PARAMETERS,
                    reason="invalid or missing parameters; repairing the step",
                    missing_parameters=error.missing_parameters,
                    error=error,
                )
            return RecoveryDecision(
                action=RecoveryAction.ASK_USER,
                reason="required information is missing and cannot be derived",
                missing_parameters=error.missing_parameters,
                error=error,
            )

        if error.category in (ErrorCategory.WORKFLOW, ErrorCategory.NOT_FOUND):
            if used("replan") < policy.max_replans:
                return RecoveryDecision(
                    action=RecoveryAction.REPLAN,
                    reason=(
                        "a prerequisite step is missing; replanning"
                        if error.category is ErrorCategory.WORKFLOW
                        else "the resource was not found; replanning to locate it"
                    ),
                    error=error,
                )
            return RecoveryDecision(
                action=RecoveryAction.FAIL, reason="replanning did not resolve the failure",
                error=error,
            )

        if error.category is ErrorCategory.BUSINESS:
            return RecoveryDecision(
                action=RecoveryAction.FAIL,
                reason="the API rejected the operation on business grounds; not retrying",
                error=error,
            )

        if used("unknown") < 1:
            return RecoveryDecision(
                action=RecoveryAction.RETRY, reason="unclassified failure; retrying once",
                error=error,
            )
        return RecoveryDecision(
            action=RecoveryAction.FAIL, reason="unclassified failure", error=error
        )

    @staticmethod
    def counter_key(action: RecoveryAction) -> str:
        return {
            RecoveryAction.WAIT_AND_RETRY: "transient",
            RecoveryAction.RETRY: "unknown",
            RecoveryAction.REFRESH_AUTH: "auth",
            RecoveryAction.FIX_PARAMETERS: "parameters",
            RecoveryAction.REPLAN: "replan",
        }.get(action, "other")

    @staticmethod
    def feedback(error: ErrorInfo, decision: RecoveryDecision) -> str:
        """Compact, prompt-safe description used when asking the model to replan."""
        payload = {
            "failed_tool": error.tool_id,
            "category": error.category.value,
            "status_code": error.status_code,
            "issues": error.issue_codes[:3],
            "message": error.message[:300],
            "missing_parameters": decision.missing_parameters,
            "guidance": decision.reason,
        }
        return json.dumps(payload, default=str)
