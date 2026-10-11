from __future__ import annotations

import math
import time
from dataclasses import dataclass

from gui_agent.agent.action_runner import ActionRunner
from gui_agent.agent.environment import DesktopEnvironment
from gui_agent.agent.feedback import (
    ActionFeedback,
    FeedbackResult,
)
from gui_agent.agent.grounding import (
    GroundingTargetNotFoundError,
)
from gui_agent.agent.observation import Observation
from gui_agent.agent.retry import RetryPolicy
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
)
from gui_agent.agent.failure_policy import (
    FailureDecision,
    FailurePhase,
    decide_recovery,
)


@dataclass(frozen=True)
class ActionTransactionResult:
    action: GUIAction
    before: Observation
    after: Observation
    feedback: FeedbackResult

    # 0 means the action did not require Grounding.
    # 1 means Grounding succeeded on the first attempt.
    # 2 means one safe Grounding retry was required.
    grounding_attempts: int = 0

    @property
    def grounding_retries(self) -> int:
        return max(
            0,
            self.grounding_attempts - 1,
        )


class ActionTransaction:
    def __init__(
        self,
        environment: DesktopEnvironment,
        runner: ActionRunner,
        feedback: ActionFeedback | None = None,
        *,
        settle_seconds: float = 0.3,
        retry_policy: RetryPolicy | None = None,
    ):
        if (
            isinstance(settle_seconds, bool)
            or not isinstance(
                settle_seconds,
                (int, float),
            )
        ):
            raise ValueError(
                "settle_seconds must be a number"
            )

        settle_seconds = float(
            settle_seconds
        )

        if not math.isfinite(
            settle_seconds
        ):
            raise ValueError(
                "settle_seconds must be finite"
            )

        if settle_seconds < 0:
            raise ValueError(
                "settle_seconds must be non-negative"
            )

        if (
            retry_policy is not None
            and not isinstance(
                retry_policy,
                RetryPolicy,
            )
        ):
            raise TypeError(
                "retry_policy must be "
                "a RetryPolicy or None"
            )

        self.environment = environment
        self.runner = runner

        self.feedback = (
            feedback
            if feedback is not None
            else ActionFeedback()
        )

        self.settle_seconds = (
            settle_seconds
        )

        self.retry_policy = (
            retry_policy
            if retry_policy is not None
            else RetryPolicy()
        )

        self.last_failure: FailureDecision | None = None

    def execute(
            self,
            action: GUIAction,
    ) -> ActionTransactionResult:
        self.last_failure = None
        phase = FailurePhase.PREFLIGHT

        try:
            requirements = self.runner.preflight(action)
            grounding_attempts = 0

            if requirements.needs_ocr:
                max_attempts = (
                    self.retry_policy.max_grounding_attempts
                )

                for attempt in range(
                        1, max_attempts + 1
                ):
                    grounding_attempts = attempt

                    phase = FailurePhase.OBSERVATION
                    before = self.environment.observe(
                        use_ocr=True
                    )

                    # run() includes both Grounding and
                    # execution. Non-grounding errors are
                    # conservatively treated as possibly
                    # occurring after dispatch.
                    phase = FailurePhase.ACTION_EXECUTION

                    try:
                        resolved_action = self.runner.run(
                            action, before
                        )
                        break


                    except GroundingTargetNotFoundError:

                        phase = FailurePhase.GROUNDING

                        if attempt >= max_attempts:
                            raise

                        phase = FailurePhase.GROUNDING
                        delay = (
                            self.retry_policy
                            .grounding_retry_delay_s
                        )

                        if delay > 0:
                            time.sleep(delay)

            else:
                phase = FailurePhase.OBSERVATION
                before = self.environment.observe(
                    use_ocr=False
                )

                phase = FailurePhase.ACTION_EXECUTION
                resolved_action = self.runner.run(
                    action, before
                )

            # The executor has returned successfully.
            # Subsequent failures must not trigger
            # another desktop action.
            phase = FailurePhase.POST_ACTION

            if (
                    resolved_action.action_type
                    != ActionType.WAIT
                    and self.settle_seconds > 0
            ):
                time.sleep(self.settle_seconds)

            after = self.environment.observe(
                use_ocr=False
            )

            feedback_result = self.feedback.compare(
                before, after
            )

            return ActionTransactionResult(
                action=resolved_action,
                before=before,
                after=after,
                feedback=feedback_result,
                grounding_attempts=grounding_attempts,
            )



        except Exception as exc:

            self.last_failure = decide_recovery(

                exc,

                phase=phase,

            )

            raise

