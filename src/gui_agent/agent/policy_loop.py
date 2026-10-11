from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gui_agent.agent.action_policy import (
    ActionPolicy,
    ActionPolicyResult,
    action_to_history_dict,
)
from gui_agent.agent.action_transaction import (
    ActionTransaction,
    ActionTransactionResult,
)
from gui_agent.agent.runtime import (
    AgentRuntime,
)
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
)


@dataclass(frozen=True)
class PolicyStepResult:
    policy: ActionPolicyResult
    transaction: ActionTransactionResult


@dataclass
class PolicyLoopResult:
    success: bool

    steps: list[
        PolicyStepResult
    ] = field(
        default_factory=list
    )

    failure_reason: str | None = None

    judge_result: (
        SuccessJudgeResult | None
    ) = None


class PolicyAgentLoop:
    def __init__(
        self,
        runtime: AgentRuntime,
        transaction: ActionTransaction,
        policy: ActionPolicy,
    ):
        self.runtime = runtime
        self.transaction = transaction
        self.policy = policy

    def run(
        self,
        *,
        success_check: Callable[
            [],
            SuccessJudgeResult,
        ],
    ) -> PolicyLoopResult:
        self.runtime.start()

        results: list[
            PolicyStepResult
        ] = []

        history: list[dict] = []

        try:
            observation = (
                self.transaction
                .environment
                .observe(
                    use_ocr=False
                )
            )

            self.runtime.update_observation(
                observation
            )

            while self.runtime.check_limits():
                prediction = (
                    self.policy.predict(
                        task=(
                            self.runtime
                            .state
                            .task
                        ),
                        observation=(
                            observation
                        ),
                        history=history,
                    )
                )

                description = (
                    f"policy: "
                    f"{prediction.action.action_type.value}"
                )

                self.runtime.begin_step(
                    description
                )

                try:
                    transaction_result = (
                        self.transaction.execute(
                            prediction.action
                        )
                    )

                except Exception as exc:
                    self.runtime.finish_step(
                        success=False,
                        error=str(exc),
                    )

                    self.runtime.mark_failed(
                        str(exc)
                    )

                    return PolicyLoopResult(
                        success=False,
                        steps=results,
                        failure_reason=str(exc),
                    )

                self.runtime.finish_step(
                    success=True
                )

                observation = (
                    transaction_result.after
                )

                self.runtime.update_observation(
                    observation
                )

                history.append(
                    action_to_history_dict(
                        transaction_result.action
                    )
                )

                results.append(
                    PolicyStepResult(
                        policy=prediction,
                        transaction=(
                            transaction_result
                        ),
                    )
                )

                # PaddleOCR and Qwen both use GPU.
                # Release Qwen before OCR success
                # verification to preserve the
                # validated Torch/Paddle coexistence
                # strategy.
                self.policy.unload()

                judge_result = (
                    success_check()
                )

                if not isinstance(
                    judge_result,
                    SuccessJudgeResult,
                ):
                    raise TypeError(
                        "success_check must "
                        "return "
                        "SuccessJudgeResult"
                    )

                if judge_result.success:
                    self.runtime.mark_done()

                    return PolicyLoopResult(
                        success=True,
                        steps=results,
                        judge_result=(
                            judge_result
                        ),
                    )

                if (
                    not self.runtime
                    .check_timeout()
                ):
                    return PolicyLoopResult(
                        success=False,
                        steps=results,
                        failure_reason=(
                            self.runtime
                            .state
                            .failure_reason
                        ),
                        judge_result=(
                            judge_result
                        ),
                    )

            return PolicyLoopResult(
                success=False,
                steps=results,
                failure_reason=(
                    self.runtime
                    .state
                    .failure_reason
                    or "agent limits exceeded"
                ),
            )

        except Exception as exc:
            self.runtime.mark_failed(
                str(exc)
            )

            return PolicyLoopResult(
                success=False,
                steps=results,
                failure_reason=str(exc),
            )

        finally:
            self.policy.unload()