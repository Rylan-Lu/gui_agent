from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gui_agent.agent.action_adapter import (
    plan_step_to_action,
)
from gui_agent.agent.action_transaction import (
    ActionTransaction,
    ActionTransactionResult,
)
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
)


@dataclass
class AgentLoopResult:
    success: bool

    transactions: list[
        ActionTransactionResult
    ] = field(
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

    def _timeout_failure(
        self,
        results: list[
            ActionTransactionResult
        ],
        *,
        judge_result: (
            SuccessJudgeResult
            | None
        ) = None,
    ) -> AgentLoopResult:
        return AgentLoopResult(
            success=False,
            transactions=results,
            failure_reason=(
                self.runtime.state.failure_reason
                or "timeout exceeded"
            ),
            judge_result=judge_result,
        )

    def run(
        self,
        plan: TaskPlan,
        *,
        final_check: Callable[
            [],
            SuccessJudgeResult,
        ]
        | None = None,
    ) -> AgentLoopResult:
        self.runtime.start()

        results: list[
            ActionTransactionResult
        ] = []

        for step in plan.steps:
            try:
                self.runtime.begin_step(
                    step.description
                )

                action = (
                    plan_step_to_action(
                        step
                    )
                )

                result = (
                    self.transaction.execute(
                        action
                    )
                )

                results.append(result)

                # Runtime should always hold
                # the latest desktop state.
                self.runtime.update_observation(
                    result.after
                )

                self.runtime.finish_step(
                    success=True
                )

                # A step may itself take longer
                # than the remaining task budget.
                if not self.runtime.check_timeout():
                    return self._timeout_failure(
                        results
                    )

            except Exception as exc:
                if self.runtime.state.history:
                    current = (
                        self.runtime
                        .state
                        .history[-1]
                    )

                    if current.success is None:
                        self.runtime.finish_step(
                            success=False,
                            error=str(exc),
                        )

                reason = str(exc)

                self.runtime.mark_failed(
                    reason
                )

                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=reason,
                )

        # Protect the transition from
        # execution into final verification.
        if not self.runtime.check_timeout():
            return self._timeout_failure(
                results
            )

        judge_result = None

        if final_check is not None:
            try:
                judge_result = (
                    final_check()
                )

                if not isinstance(
                    judge_result,
                    SuccessJudgeResult,
                ):
                    raise TypeError(
                        "final_check must return "
                        "SuccessJudgeResult"
                    )

            except Exception as exc:
                reason = (
                    "success judge failed: "
                    f"{exc}"
                )

                self.runtime.mark_failed(
                    reason
                )

                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=reason,
                )

            # Success Judge execution is also
            # part of the total task time budget.
            if not self.runtime.check_timeout():
                return self._timeout_failure(
                    results,
                    judge_result=judge_result,
                )

            if not judge_result.success:
                self.runtime.mark_failed(
                    judge_result.reason
                )

                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=(
                        judge_result.reason
                    ),
                    judge_result=judge_result,
                )

        self.runtime.mark_done()

        return AgentLoopResult(
            success=True,
            transactions=results,
            judge_result=judge_result,
        )