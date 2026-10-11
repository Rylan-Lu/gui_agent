
from unittest.mock import Mock

import pytest

from gui_agent.agent.action_transaction import (
    ActionTransaction,
)
from gui_agent.agent.failure_policy import (
    FailurePhase,
    RecoveryDirective,
)
from gui_agent.agent.grounding import (
    GroundingTargetNotFoundError,
)
from gui_agent.agent.retry import RetryPolicy
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
)


def make_transaction(
    *,
    needs_ocr=False,
    retry_policy=None,
):
    environment = Mock()
    runner = Mock()
    feedback = Mock()

    runner.preflight.return_value.needs_ocr = needs_ocr

    transaction = ActionTransaction(
        environment=environment,
        runner=runner,
        feedback=feedback,
        settle_seconds=0,
        retry_policy=retry_policy,
    )

    action = GUIAction(
        action_type=ActionType.TYPE_TEXT,
        text="hello",
    )

    runner.run.return_value = action

    return transaction, environment, runner, action


def test_preflight_failure_recorded():
    tx, environment, runner, action = (
        make_transaction()
    )

    runner.preflight.side_effect = ValueError(
        "invalid action"
    )

    with pytest.raises(ValueError):
        tx.execute(action)

    assert tx.last_failure.phase == FailurePhase.PREFLIGHT
    assert tx.last_failure.directive == RecoveryDirective.STOP

    environment.observe.assert_not_called()


def test_grounding_failure_requests_replan():
    tx, environment, runner, action = make_transaction(
        needs_ocr=True,
        retry_policy=RetryPolicy(
            max_grounding_attempts=2,
            grounding_retry_delay_s=0,
        ),
    )

    runner.run.side_effect = (
        GroundingTargetNotFoundError("not found")
    )

    with pytest.raises(GroundingTargetNotFoundError):
        tx.execute(action)

    assert tx.last_failure.phase == FailurePhase.GROUNDING
    assert tx.last_failure.directive == RecoveryDirective.REPLAN
    assert tx.last_failure.action_may_have_executed is False

    assert runner.run.call_count == 2


def test_execution_failure_requires_verification():
    tx, environment, runner, action = (
        make_transaction()
    )

    runner.run.side_effect = RuntimeError(
        "execution failed"
    )

    with pytest.raises(RuntimeError):
        tx.execute(action)

    assert (
        tx.last_failure.phase
        == FailurePhase.ACTION_EXECUTION
    )
    assert (
        tx.last_failure.directive
        == RecoveryDirective.VERIFY_STATE
    )


def test_post_action_failure_requires_verification():
    tx, environment, runner, action = (
        make_transaction()
    )

    environment.observe.side_effect = [
        Mock(),
        RuntimeError("screenshot failed"),
    ]

    with pytest.raises(RuntimeError):
        tx.execute(action)

    assert (
        tx.last_failure.phase
        == FailurePhase.POST_ACTION
    )
    assert (
        tx.last_failure.directive
        == RecoveryDirective.VERIFY_STATE
    )
    assert runner.run.call_count == 1


def test_success_clears_previous_failure():
    tx, environment, runner, action = (
        make_transaction()
    )

    runner.run.side_effect = [
        RuntimeError("temporary failure"),
        action,
    ]

    with pytest.raises(RuntimeError):
        tx.execute(action)

    assert tx.last_failure is not None

    result = tx.execute(action)

    assert result.action == action
    assert tx.last_failure is None


def test_observation_error_is_not_misclassified_as_grounding():
    tx, environment, runner, action = (
        make_transaction(needs_ocr=True)
    )

    environment.observe.side_effect = (
        GroundingTargetNotFoundError(
            "observation failed"
        )
    )

    with pytest.raises(GroundingTargetNotFoundError):
        tx.execute(action)

    assert tx.last_failure.phase == (
        FailurePhase.OBSERVATION
    )
    assert tx.last_failure.directive == (
        RecoveryDirective.STOP
    )

    runner.run.assert_not_called()


def test_unexpected_grounding_error_during_dispatch_is_unsafe():
    tx, environment, runner, action = (
        make_transaction(needs_ocr=False)
    )

    runner.run.side_effect = (
        GroundingTargetNotFoundError(
            "unexpected dispatch failure"
        )
    )

    with pytest.raises(GroundingTargetNotFoundError):
        tx.execute(action)

    assert tx.last_failure.phase == (
        FailurePhase.ACTION_EXECUTION
    )
    assert tx.last_failure.directive == (
        RecoveryDirective.VERIFY_STATE
    )

    assert tx.last_failure.action_may_have_executed
