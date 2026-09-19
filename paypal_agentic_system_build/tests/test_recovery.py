"""Step 9 verification: error classification and bounded recovery strategy."""

from __future__ import annotations

from app.recovery.handler import (
    ErrorCategory,
    RecoveryAction,
    RecoveryHandler,
    RecoveryPolicy,
    classify,
)
from app.tools.models import ExecutionResult


def failure(**kwargs) -> ExecutionResult:
    return ExecutionResult(tool_id="payments.refund_captured_payment", success=False, **kwargs)


# ------------------------------------------------------------ classification --
def test_timeouts_and_network_errors_are_transient():
    assert classify(failure(error_type="timeout", error="too slow")).category is ErrorCategory.TRANSIENT
    assert classify(failure(error_type="network", error="reset")).category is ErrorCategory.TRANSIENT


def test_server_errors_are_transient_and_retryable():
    info = classify(failure(status_code=503, error_type="http", error="unavailable"))
    assert info.category is ErrorCategory.TRANSIENT and info.retryable


def test_rate_limit_captures_retry_after():
    info = classify(
        failure(
            status_code=429,
            error_type="http",
            error="too many requests",
            response_headers={"retry-after": "3"},
        )
    )
    assert info.category is ErrorCategory.RATE_LIMIT
    assert info.retry_after == 3.0


def test_authentication_and_permission_are_distinguished():
    assert classify(failure(status_code=401, error_type="http")).category is ErrorCategory.AUTHENTICATION
    assert classify(failure(status_code=403, error_type="http")).category is ErrorCategory.PERMISSION


def test_not_found_is_its_own_category():
    assert classify(failure(status_code=404, error_type="http")).category is ErrorCategory.NOT_FOUND


def test_missing_parameters_are_extracted_from_the_error_body():
    info = classify(
        failure(
            status_code=422,
            error_type="http",
            error="The requested action could not be performed",
            data={
                "name": "UNPROCESSABLE_ENTITY",
                "details": [{"issue": "MISSING_REQUIRED_PARAMETER", "field": "/amount/value"}],
            },
        )
    )
    assert info.category is ErrorCategory.VALIDATION
    assert info.missing_parameters == ["value"]


def test_local_validation_failures_are_classified_without_a_status_code():
    info = classify(
        failure(error_type="validation", error="missing required path parameter(s): capture_id")
    )
    assert info.category is ErrorCategory.VALIDATION
    assert "capture_id" in info.missing_parameters


def test_precondition_issues_are_workflow_errors():
    info = classify(
        failure(
            status_code=422,
            error_type="http",
            error="cannot refund",
            data={"details": [{"issue": "AUTHORIZATION_NOT_CAPTURED"}]},
        )
    )
    assert info.category is ErrorCategory.WORKFLOW


def test_already_done_issues_are_business_errors():
    info = classify(
        failure(
            status_code=422,
            error_type="http",
            error="already refunded",
            data={"details": [{"issue": "CAPTURE_FULLY_REFUNDED"}]},
        )
    )
    assert info.category is ErrorCategory.BUSINESS


# ----------------------------------------------------------------- strategy --
def test_transient_failures_retry_with_exponential_backoff():
    handler = RecoveryHandler()
    error = classify(failure(error_type="timeout", error="slow"))

    first = handler.decide(error, {})
    second = handler.decide(error, {"transient": 1})
    third = handler.decide(error, {"transient": 2})
    exhausted = handler.decide(error, {"transient": 3})

    assert first.action is RecoveryAction.WAIT_AND_RETRY
    assert first.delay_seconds < second.delay_seconds < third.delay_seconds
    assert exhausted.action is RecoveryAction.FAIL


def test_backoff_is_capped():
    policy = RecoveryPolicy(base_delay_seconds=1, max_delay_seconds=4)
    assert policy.backoff(10) == 4


def test_rate_limit_respects_retry_after_over_backoff():
    handler = RecoveryHandler()
    error = classify(
        failure(status_code=429, error_type="http", response_headers={"retry-after": "5"})
    )
    decision = handler.decide(error, {})
    assert decision.action is RecoveryAction.WAIT_AND_RETRY
    assert decision.delay_seconds == 5


def test_authentication_refreshes_once_then_fails():
    handler = RecoveryHandler()
    error = classify(failure(status_code=401, error_type="http"))
    assert handler.decide(error, {}).action is RecoveryAction.REFRESH_AUTH
    assert handler.decide(error, {"auth": 1}).action is RecoveryAction.FAIL


def test_validation_repairs_once_then_asks_the_user():
    handler = RecoveryHandler()
    error = classify(
        failure(error_type="validation", error="missing required path parameter(s): capture_id")
    )
    assert handler.decide(error, {}).action is RecoveryAction.FIX_PARAMETERS
    escalated = handler.decide(error, {"parameters": 1})
    assert escalated.action is RecoveryAction.ASK_USER
    assert "capture_id" in escalated.missing_parameters


def test_workflow_errors_replan_and_business_errors_do_not():
    handler = RecoveryHandler()
    workflow = classify(
        failure(status_code=422, error_type="http", data={"details": [{"issue": "ORDER_NOT_APPROVED"}]})
    )
    business = classify(
        failure(status_code=422, error_type="http", data={"details": [{"issue": "ALREADY_CAPTURED"}]})
    )
    assert handler.decide(workflow, {}).action is RecoveryAction.REPLAN
    assert handler.decide(business, {}).action is RecoveryAction.FAIL


def test_permission_errors_are_never_retried():
    handler = RecoveryHandler()
    decision = handler.decide(classify(failure(status_code=403, error_type="http")), {})
    assert decision.action is RecoveryAction.FAIL and decision.terminal


def test_feedback_is_prompt_safe_and_actionable():
    handler = RecoveryHandler()
    error = classify(
        failure(status_code=422, error_type="http", data={"details": [{"issue": "ORDER_NOT_APPROVED"}]})
    )
    feedback = RecoveryHandler.feedback(error, handler.decide(error, {}))
    assert "ORDER_NOT_APPROVED" in feedback
    assert "secret" not in feedback.lower() and "bearer" not in feedback.lower()
