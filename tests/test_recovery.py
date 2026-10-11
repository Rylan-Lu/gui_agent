
from unittest.mock import Mock

from gui_agent.agent.failure_policy import (
    FailurePhase,
    decide_recovery,
)
from gui_agent.agent.grounding import (
    GroundingTargetNotFoundError,
)
from gui_agent.agent.recovery import (
    RecoveryEngine,
    RecoveryStatus,
)
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
)


def make_engine():
    environment = Mock()
    observation = Mock()
    environment.observe.return_value = observation

    return (
        RecoveryEngine(environment),
        environment,
        observation,
    )


def execution_failure():
    return decide_recovery(
        RuntimeError("action failed"),
        phase=FailurePhase.ACTION_EXECUTION,
    )


def test_stop_does_not_observe():
    engine, env, _ = make_engine()

    decision = decide_recovery(
        ValueError("invalid action"),
        phase=FailurePhase.PREFLIGHT,
    )

    result = engine.evaluate(decision=decision)

    assert result.status == RecoveryStatus.STOPPED
    env.observe.assert_not_called()


def test_grounding_failure_requests_replan():
    engine, env, _ = make_engine()

    decision = decide_recovery(
        GroundingTargetNotFoundError("missing"),
        phase=FailurePhase.GROUNDING,
    )

    result = engine.evaluate(decision=decision)

    assert result.status == RecoveryStatus.NEEDS_REPLAN
    env.observe.assert_not_called()


def test_execution_failure_can_be_verified():
    engine, env, observation = make_engine()

    judge = Mock(
        return_value=SuccessJudgeResult(
            success=True,
            reason="expected text found",
        )
    )

    result = engine.evaluate(
        decision=execution_failure(),
        judge=judge,
    )

    assert result.status == RecoveryStatus.JUDGE_PASSED
    assert result.judge_result.success is True
    assert result.observation is observation

    env.observe.assert_called_once_with(use_ocr=True)
    judge.assert_called_once_with(observation)


def test_judge_failure_does_not_retry_action():
    engine, env, observation = make_engine()

    judge = Mock(
        return_value=SuccessJudgeResult(
            success=False,
            reason="text missing",
        )
    )

    result = engine.evaluate(
        decision=execution_failure(),
        judge=judge,
    )

    assert result.status == RecoveryStatus.UNRESOLVED
    assert result.judge_result.success is False

    env.observe.assert_called_once()
    judge.assert_called_once_with(observation)


def test_missing_judge_returns_unresolved():
    engine, env, _ = make_engine()

    result = engine.evaluate(
        decision=execution_failure(),
    )

    assert result.status == RecoveryStatus.UNRESOLVED
    env.observe.assert_not_called()


def test_observation_exception_is_handled():
    engine, env, _ = make_engine()

    env.observe.side_effect = RuntimeError(
        "screenshot failure"
    )
    judge = Mock()

    result = engine.evaluate(
        decision=execution_failure(),
        judge=judge,
    )

    assert result.status == RecoveryStatus.UNRESOLVED
    assert "RuntimeError" in result.reason
    judge.assert_not_called()


def test_judge_exception_is_handled():
    engine, _, observation = make_engine()

    judge = Mock(
        side_effect=RuntimeError("OCR failure")
    )

    result = engine.evaluate(
        decision=execution_failure(),
        judge=judge,
    )

    assert result.status == RecoveryStatus.UNRESOLVED
    assert result.observation is observation


def test_invalid_judge_result_is_rejected():
    engine, _, _ = make_engine()

    judge = Mock(return_value=True)

    result = engine.evaluate(
        decision=execution_failure(),
        judge=judge,
    )

    assert result.status == RecoveryStatus.UNRESOLVED
    assert "TypeError" in result.reason
