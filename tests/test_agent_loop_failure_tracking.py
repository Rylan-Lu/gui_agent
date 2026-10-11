
from types import SimpleNamespace
from unittest.mock import Mock

from gui_agent.agent.failure_policy import (
    FailurePhase,
    RecoveryDirective,
    decide_recovery,
)
from gui_agent.agent.grounding import (
    GroundingTargetNotFoundError,
)
from gui_agent.agent.loop import AgentLoop
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
)


def make_loop(monkeypatch):
    action = GUIAction(
        action_type=ActionType.TYPE_TEXT,
        text="hello",
    )

    monkeypatch.setattr(
        "gui_agent.agent.loop.plan_step_to_action",
        lambda step: action,
    )

    runtime = Mock()
    runtime.state.history = [
        SimpleNamespace(success=None)
    ]
    runtime.state.failure_reason = None
    runtime.check_timeout.return_value = True

    transaction = Mock()
    transaction.last_failure = None
    transaction.execute.return_value = (
        SimpleNamespace(after=object())
    )

    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(description="test step")
        ]
    )

    return (
        AgentLoop(runtime, transaction),
        runtime,
        transaction,
        plan,
    )


def test_grounding_failure_decision_propagated(
    monkeypatch,
):
    loop, runtime, tx, plan = make_loop(
        monkeypatch
    )

    error = GroundingTargetNotFoundError(
        "target missing"
    )
    tx.execute.side_effect = error
    tx.last_failure = decide_recovery(
        error,
        phase=FailurePhase.GROUNDING,
    )

    result = loop.run(plan)

    assert result.success is False
    assert result.failure_decision.phase == (
        FailurePhase.GROUNDING
    )
    assert result.failure_decision.directive == (
        RecoveryDirective.REPLAN
    )
    assert (
        result.failure_decision.action_may_have_executed
        is False
    )


def test_execution_failure_decision_propagated(
    monkeypatch,
):
    loop, runtime, tx, plan = make_loop(
        monkeypatch
    )

    error = RuntimeError("dispatch failed")
    tx.execute.side_effect = error
    tx.last_failure = decide_recovery(
        error,
        phase=FailurePhase.ACTION_EXECUTION,
    )

    result = loop.run(plan)

    assert result.success is False
    assert result.failure_decision.directive == (
        RecoveryDirective.VERIFY_STATE
    )


def test_missing_transaction_record_is_conservative(
    monkeypatch,
):
    loop, runtime, tx, plan = make_loop(
        monkeypatch
    )

    tx.execute.side_effect = RuntimeError(
        "unknown execution state"
    )

    result = loop.run(plan)

    assert result.failure_decision.phase == (
        FailurePhase.ACTION_EXECUTION
    )
    assert result.failure_decision.directive == (
        RecoveryDirective.VERIFY_STATE
    )


def test_adapter_failure_does_not_use_stale_record(
    monkeypatch,
):
    loop, runtime, tx, plan = make_loop(
        monkeypatch
    )

    def broken_adapter(step):
        raise ValueError("invalid plan step")

    monkeypatch.setattr(
        "gui_agent.agent.loop.plan_step_to_action",
        broken_adapter,
    )

    old_error = GroundingTargetNotFoundError(
        "previous failure"
    )
    tx.last_failure = decide_recovery(
        old_error,
        phase=FailurePhase.GROUNDING,
    )

    result = loop.run(plan)

    assert result.failure_decision.phase == (
        FailurePhase.PREFLIGHT
    )
    assert result.failure_decision.directive == (
        RecoveryDirective.STOP
    )
    tx.execute.assert_not_called()


def test_runtime_failure_after_action_requires_check(
    monkeypatch,
):
    loop, runtime, tx, plan = make_loop(
        monkeypatch
    )

    runtime.update_observation.side_effect = (
        RuntimeError("runtime update failed")
    )

    result = loop.run(plan)

    assert result.failure_decision.phase == (
        FailurePhase.POST_ACTION
    )
    assert result.failure_decision.directive == (
        RecoveryDirective.VERIFY_STATE
    )
    tx.execute.assert_called_once()


def test_judge_exception_requires_verification(
    monkeypatch,
):
    loop, runtime, tx, plan = make_loop(
        monkeypatch
    )

    def broken_judge():
        raise RuntimeError("OCR failed")

    result = loop.run(
        plan,
        final_check=broken_judge,
    )

    assert result.success is False
    assert result.failure_decision.phase == (
        FailurePhase.SUCCESS_JUDGE
    )
    assert result.failure_decision.directive == (
        RecoveryDirective.VERIFY_STATE
    )


def test_success_has_no_failure_decision(
    monkeypatch,
):
    loop, runtime, tx, plan = make_loop(
        monkeypatch
    )

    result = loop.run(plan)

    assert result.success is True
    assert result.failure_decision is None
    runtime.mark_done.assert_called_once()
