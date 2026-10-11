
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gui_agent.agent.grounding import (
    GroundingTargetNotFoundError,
)


class FailurePhase(str, Enum):
    POLICY = "policy"
    PREFLIGHT = "preflight"
    GROUNDING = "grounding"
    ACTION_EXECUTION = "action_execution"
    POST_ACTION = "post_action"
    SUCCESS_JUDGE = "success_judge"
    OBSERVATION = "observation"


class RecoveryDirective(str, Enum):
    REPLAN = "replan"
    VERIFY_STATE = "verify_state"
    STOP = "stop"


@dataclass(frozen=True)
class FailureDecision:
    phase: FailurePhase
    directive: RecoveryDirective
    action_may_have_executed: bool
    exception_type: str


def decide_recovery(
    error: Exception,
    *,
    phase: FailurePhase,
) -> FailureDecision:
    """
    Classify a failure without executing recovery.

    The caller must supply the verified failure phase.
    Never infer execution safety from exception type alone.
    """
    if not isinstance(error, Exception):
        raise TypeError("error must be an Exception")

    if not isinstance(phase, FailurePhase):
        raise TypeError("phase must be a FailurePhase")

    if (
        phase is FailurePhase.GROUNDING
        and isinstance(
            error,
            GroundingTargetNotFoundError,
        )
    ):
        # Grounding attempts have already been exhausted.
        # No desktop action should have been dispatched.
        directive = RecoveryDirective.REPLAN
        may_have_executed = False

    elif phase in {
        FailurePhase.ACTION_EXECUTION,
        FailurePhase.POST_ACTION,
        FailurePhase.SUCCESS_JUDGE,
    }:
        # Side effects may already exist.
        # Observe and verify before doing anything else.
        directive = RecoveryDirective.VERIFY_STATE
        may_have_executed = True

    else:
        # Unknown or unsupported failures stop safely.
        directive = RecoveryDirective.STOP
        may_have_executed = False

    return FailureDecision(
        phase=phase,
        directive=directive,
        action_may_have_executed=may_have_executed,
        exception_type=type(error).__name__,
    )
