
import pytest

from gui_agent.agent.failure_policy import (
    FailurePhase,
    RecoveryDirective,
    decide_recovery,
)
from gui_agent.agent.grounding import (
    GroundingTargetNotFoundError,
)


def test_grounding_failure_requests_replan():
    result = decide_recovery(
        GroundingTargetNotFoundError("not found"),
        phase=FailurePhase.GROUNDING,
    )

    assert result.directive == RecoveryDirective.REPLAN
    assert result.action_may_have_executed is False


@pytest.mark.parametrize(
    "phase",
    [
        FailurePhase.ACTION_EXECUTION,
        FailurePhase.POST_ACTION,
        FailurePhase.SUCCESS_JUDGE,
    ],
)
def test_possible_side_effect_requires_verification(phase):
    result = decide_recovery(
        RuntimeError("failure"),
        phase=phase,
    )

    assert result.directive == RecoveryDirective.VERIFY_STATE
    assert result.action_may_have_executed is True


def test_unknown_grounding_failure_stops():
    result = decide_recovery(
        ValueError("unexpected"),
        phase=FailurePhase.GROUNDING,
    )

    assert result.directive == RecoveryDirective.STOP


def test_policy_failure_stops():
    result = decide_recovery(
        RuntimeError("bad model output"),
        phase=FailurePhase.POLICY,
    )

    assert result.directive == RecoveryDirective.STOP


def test_invalid_phase_rejected():
    with pytest.raises(TypeError):
        decide_recovery(
            RuntimeError("failure"),
            phase="grounding",
        )
