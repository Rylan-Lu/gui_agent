from __future__ import annotations

import time
from dataclasses import dataclass

from gui_agent.agent.action_runner import ActionRunner
from gui_agent.agent.feedback import ActionFeedback, FeedbackResult
from gui_agent.agent.observation import Observation
from gui_agent.datasets.schema import ActionType, GUIAction


@dataclass(frozen=True)
class ActionTransactionResult:
    action: GUIAction
    before: Observation
    after: Observation
    feedback: FeedbackResult


class ActionTransaction:
    def __init__(
        self,
        environment,
        runner: ActionRunner,
        feedback: ActionFeedback | None = None,
        *,
        settle_seconds: float = 0.3,
    ):
        if settle_seconds < 0:
            raise ValueError("settle_seconds must be non-negative")

        self.environment = environment
        self.runner = runner
        self.feedback = feedback or ActionFeedback()
        self.settle_seconds = settle_seconds

    def execute(
        self,
        action: GUIAction,
    ) -> ActionTransactionResult:

        needs_ocr = (
            action.action_type
            in {ActionType.CLICK, ActionType.DOUBLE_CLICK}
            and action.position is None
        )

        before = self.environment.observe(
            use_ocr=needs_ocr
        )

        resolved_action = self.runner.run(
            action,
            before,
        )

        if self.settle_seconds > 0:
            time.sleep(self.settle_seconds)

        after = self.environment.observe(
            use_ocr=False
        )

        feedback_result = self.feedback.compare(
            before,
            after,
        )

        return ActionTransactionResult(
            action=resolved_action,
            before=before,
            after=after,
            feedback=feedback_result,
        )