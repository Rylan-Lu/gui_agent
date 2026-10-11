
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from gui_agent.agent.environment import (
    DesktopEnvironment,
)
from gui_agent.agent.failure_policy import (
    FailureDecision,
    RecoveryDirective,
)
from gui_agent.agent.observation import Observation
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
)


class RecoveryStatus(str, Enum):
    JUDGE_PASSED = "judge_passed"
    UNRESOLVED = "unresolved"
    NEEDS_REPLAN = "needs_replan"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RecoveryResult:
    status: RecoveryStatus
    decision: FailureDecision
    reason: str
    observation: Observation | None = None
    judge_result: SuccessJudgeResult | None = None


class RecoveryEngine:
    """
    Evaluate recovery decisions without replaying actions.

    Never dispatches GUI actions itself.
    """

    def __init__(
        self,
        environment: DesktopEnvironment,
    ):
        self.environment = environment

    def evaluate(
        self,
        *,
        decision: FailureDecision,
        judge: Callable[
            [Observation],
            SuccessJudgeResult,
        ] | None = None,
    ) -> RecoveryResult:

        if not isinstance(decision, FailureDecision):
            raise TypeError(
                "decision must be FailureDecision"
            )

        if decision.directive == RecoveryDirective.STOP:
            return RecoveryResult(
                status=RecoveryStatus.STOPPED,
                decision=decision,
                reason="recovery not permitted",
            )

        if decision.directive == RecoveryDirective.REPLAN:
            return RecoveryResult(
                status=RecoveryStatus.NEEDS_REPLAN,
                decision=decision,
                reason="new plan required",
            )

        # VERIFY_STATE:
        # We need an explicit completion judge.
        if judge is None:
            return RecoveryResult(
                status=RecoveryStatus.UNRESOLVED,
                decision=decision,
                reason="no verification judge available",
            )

        # Obtain a fresh OCR observation.
        try:
            observation = self.environment.observe(
                use_ocr=True
            )
        except Exception as exc:
            return RecoveryResult(
                status=RecoveryStatus.UNRESOLVED,
                decision=decision,
                reason=(
                    "recovery observation failed: "
                    f"{type(exc).__name__}"
                ),
            )

        try:
            judge_result = judge(observation)

            if not isinstance(
                judge_result,
                SuccessJudgeResult,
            ):
                raise TypeError(
                    "judge must return SuccessJudgeResult"
                )

        except Exception as exc:
            return RecoveryResult(
                status=RecoveryStatus.UNRESOLVED,
                decision=decision,
                reason=(
                    "recovery judge failed: "
                    f"{type(exc).__name__}"
                ),
                observation=observation,
            )

        if judge_result.success:
            return RecoveryResult(
                status=RecoveryStatus.JUDGE_PASSED,
                decision=decision,
                reason="success judge passed",
                observation=observation,
                judge_result=judge_result,
            )

        return RecoveryResult(
            status=RecoveryStatus.UNRESOLVED,
            decision=decision,
            reason="task completion not confirmed",
            observation=observation,
            judge_result=judge_result,
        )
