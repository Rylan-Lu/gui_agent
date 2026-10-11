
from types import SimpleNamespace
from unittest.mock import Mock

from gui_agent.agent.failure_policy import (
    FailurePhase,
    decide_recovery,
)
from gui_agent.agent.grounding import (
    GroundingTargetNotFoundError,
)
from gui_agent.agent.loop import AgentLoop
from gui_agent.agent.recovery import (
    RecoveryEngine,
    RecoveryStatus,
)
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
)
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
)


def make_loop(
    monkeypatch,
    *,
    error=None,
    phase=FailurePhase.ACTION_EXECUTION,
):
    action = GUIAction(
        action_type=ActionType.TYPE_TEXT,
        text="hello",
    )

    monkeypatch.setattr(
        "gui_agent.agent.loop.plan_step_to_action",
        lambda step: action,
    )

    runtime = AgentRuntime.create(
        task="test recovery",
        max_steps=3,
        timeout_s=60,
    )

    error = error or RuntimeError("action failed")

    transaction = Mock()
    transaction.execute.side_effect = error
    transaction.last_failure = decide_recovery(
        error,
        phase=phase,
    )

    environment = Mock()
    observation = object()
    environment.observe.return_value = observation

    engine = RecoveryEngine(environment)

    loop = AgentLoop(
        runtime=runtime,
        transaction=transaction,
        recovery_engine=engine,
    )

    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(
                description="test action"
            )
        ]
    )

    return (
        loop,
        runtime,
        transaction,
        environment,
        observation,
        plan,
    )


def test_recovery_judge_passed_marks_task_done(
    monkeypatch,
):
    loop, runtime, tx, env, obs, plan = (
        make_loop(monkeypatch)
    )

    judge = Mock(
        return_value=SuccessJudgeResult(
            success=True,
            reason="task verified",
        )
    )

    result = loop.run(
        plan,
        recovery_check=judge,
    )

    assert result.success is True
    assert result.recovery_result.status == (
        RecoveryStatus.JUDGE_PASSED
    )
    assert runtime.state.done is True
    assert runtime.state.failed is False
    assert runtime.state.observation is obs

    # The failed GUI action is not replayed.
    tx.execute.assert_called_once()
    env.observe.assert_called_once_with(
        use_ocr=True
    )
    judge.assert_called_once_with(obs)


def test_unresolved_recovery_stops_safely(
    monkeypatch,
):
    loop, runtime, tx, env, obs, plan = (
        make_loop(monkeypatch)
    )

    judge = Mock(
        return_value=SuccessJudgeResult(
            success=False,
            reason="task not completed",
        )
    )

    result = loop.run(
        plan,
        recovery_check=judge,
    )

    assert result.success is False
    assert result.recovery_result.status == (
        RecoveryStatus.UNRESOLVED
    )
    assert runtime.state.failed is True

    tx.execute.assert_called_once()


def test_grounding_failure_requests_replan(
    monkeypatch,
):
    error = GroundingTargetNotFoundError(
        "target missing"
    )

    loop, runtime, tx, env, obs, plan = (
        make_loop(
            monkeypatch,
            error=error,
            phase=FailurePhase.GROUNDING,
        )
    )

    result = loop.run(plan)

    assert result.success is False
    assert result.recovery_result.status == (
        RecoveryStatus.NEEDS_REPLAN
    )
    assert runtime.state.failed is True

    env.observe.assert_not_called()
    tx.execute.assert_called_once()


def test_missing_recovery_judge_does_not_verify(
    monkeypatch,
):
    loop, runtime, tx, env, obs, plan = (
        make_loop(monkeypatch)
    )

    result = loop.run(plan)

    assert result.success is False
    assert result.recovery_result.status == (
        RecoveryStatus.UNRESOLVED
    )

    env.observe.assert_not_called()
    tx.execute.assert_called_once()


def test_timeout_during_recovery_prevents_success(
    monkeypatch,
):
    loop, runtime, tx, env, obs, plan = (
        make_loop(monkeypatch)
    )

    def judge(observation):
        # Simulate that the recovery verification
        # consumed more than the remaining budget.
        runtime.state.started_at -= 1000.0

        return SuccessJudgeResult(
            success=True,
            reason="task verified too late",
        )

    result = loop.run(
        plan,
        recovery_check=judge,
    )

    assert result.success is False
    assert runtime.state.done is False
    assert runtime.state.failed is True
    assert "timeout exceeded" in (
        result.failure_reason
    )

    tx.execute.assert_called_once()


def test_recovery_engine_exception_fails_closed(
    monkeypatch,
):
    loop, runtime, tx, env, obs, plan = (
        make_loop(monkeypatch)
    )

    broken_engine = Mock()
    broken_engine.evaluate.side_effect = (
        RuntimeError("recovery unavailable")
    )

    loop.recovery_engine = broken_engine

    result = loop.run(plan)

    assert result.success is False
    assert runtime.state.failed is True
    assert "recovery evaluation failed" in (
        result.failure_reason
    )

    tx.execute.assert_called_once()
