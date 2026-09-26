from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gui_agent.agent.action_adapter import plan_step_to_action
from gui_agent.agent.action_transaction import (
    ActionTransaction,
    ActionTransactionResult,
)
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import SuccessJudgeResult


@dataclass
class AgentLoopResult:
    success: bool
    transactions: list[ActionTransactionResult] = field(
        default_factory=list
    )
    failure_reason: str | None = None
    judge_result: SuccessJudgeResult | None = None


class AgentLoop:
    def __init__(
        self,
        runtime: AgentRuntime,
        transaction: ActionTransaction,
    ):
        self.runtime = runtime
        self.transaction = transaction

    def run(
        self,
        plan: TaskPlan,
        *,
        final_check: Callable[[], SuccessJudgeResult] | None = None,
    ) -> AgentLoopResult:
        self.runtime.start()

        results = []

        for step in plan.steps:
            try:
                self.runtime.begin_step(step.description)

                action = plan_step_to_action(step)

                result = self.transaction.execute(action)
                results.append(result)

                self.runtime.finish_step(success=True)

            except Exception as exc:
                if self.runtime.state.history:
                    current = self.runtime.state.history[-1]

                    if current.success is None:
                        self.runtime.finish_step(
                            success=False,
                            error=str(exc),
                        )

                self.runtime.mark_failed(str(exc))

                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=str(exc),
                )

        judge_result = None

        if final_check is not None:
            try:
                judge_result = final_check()
            except Exception as exc:
                self.runtime.mark_failed(
                    f"success judge failed: {exc}"
                )

                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=str(exc),
                )

            if not judge_result.success:
                self.runtime.mark_failed(
                    judge_result.reason
                )

                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=judge_result.reason,
                    judge_result=judge_result,
                )

        self.runtime.mark_done()

        return AgentLoopResult(
            success=True,
            transactions=results,
            judge_result=judge_result,
        )