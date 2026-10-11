from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gui_agent.agent.action_adapter import plan_step_to_action
from gui_agent.agent.action_transaction import (
    ActionTransaction,
    ActionTransactionResult,
)
from gui_agent.agent.failure_policy import (
    FailureDecision,
    FailurePhase,
    decide_recovery,
)
from gui_agent.agent.observation import Observation
from gui_agent.agent.plan_schema import PlanStep, TaskPlan
from gui_agent.agent.recovery import (
    RecoveryEngine,
    RecoveryResult,
    RecoveryStatus,
)
from gui_agent.agent.replan import (
    ReplanContext,
    Replanner,
    temporary_replan_screenshot,
)
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import SuccessJudgeResult

from gui_agent.agent.action_adapter import plan_step_to_action
from gui_agent.agent.action_policy import ActionPolicy

import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopResult:
    success: bool
    transactions: list[ActionTransactionResult] = field(default_factory=list)
    failure_reason: str | None = None
    judge_result: SuccessJudgeResult | None = None
    failure_decision: FailureDecision | None = None
    recovery_result: RecoveryResult | None = None


class AgentLoop:
    def __init__(
        self,
        runtime: AgentRuntime,
        transaction: ActionTransaction,
        recovery_engine: RecoveryEngine | None = None,
        *,
        replanner: Replanner | None = None,
        action_policy: ActionPolicy | None = None,
    ):
        self.runtime = runtime
        self.transaction = transaction
        self.recovery_engine = recovery_engine
        self.replanner = replanner
        self.action_policy = (
            action_policy
            if action_policy is not None
            else ActionPolicy()
        )

    def _timeout_failure(
        self,
        results: list[ActionTransactionResult],
        *,
        judge_result: SuccessJudgeResult | None = None,
        failure_decision: FailureDecision | None = None,
        recovery_result: RecoveryResult | None = None,
    ) -> AgentLoopResult:
        return AgentLoopResult(
            success=False,
            transactions=results,
            failure_reason=(
                self.runtime.state.failure_reason or "timeout exceeded"
            ),
            judge_result=judge_result,
            failure_decision=failure_decision,
            recovery_result=recovery_result,
        )

    def run(
        self,
        plan: TaskPlan,
        *,
        final_check: Callable[[], SuccessJudgeResult] | None = None,
        recovery_check: Callable[[Observation], SuccessJudgeResult] | None = None,
    ) -> AgentLoopResult:
        self.runtime.start()
        results: list[ActionTransactionResult] = []
        pending_steps: list[PlanStep] = list(plan.steps)
        completed_steps: list[PlanStep] = []
        replan_count = 0

        while pending_steps:
            step = pending_steps.pop(0)
            transaction_invoked = False
            transaction_returned = False

            try:
                self.runtime.begin_step(step.description)
                action = plan_step_to_action(step)
                transaction_invoked = True
                result = self.transaction.execute(action)
                transaction_returned = True
                results.append(result)

                # Runtime's latest screen must match the successful transaction.
                self.runtime.update_observation(result.after)
                self.runtime.finish_step(success=True)
                completed_steps.append(step)

                if not self.runtime.check_timeout():
                    return self._timeout_failure(results)

            except Exception as exc:
                if self.runtime.state.history:
                    current = self.runtime.state.history[-1]
                    if current.success is None:
                        self.runtime.finish_step(success=False, error=str(exc))

                if transaction_returned:
                    failure_decision = decide_recovery(
                        exc, phase=FailurePhase.POST_ACTION
                    )
                elif transaction_invoked:
                    recorded = getattr(self.transaction, "last_failure", None)
                    if isinstance(recorded, FailureDecision):
                        failure_decision = recorded
                    else:
                        # When the execution phase is unknown, assume desktop
                        # side effects may already have occurred.
                        failure_decision = decide_recovery(
                            exc, phase=FailurePhase.ACTION_EXECUTION
                        )
                else:
                    failure_decision = decide_recovery(
                        exc, phase=FailurePhase.PREFLIGHT
                    )

                reason = str(exc)
                recovery_result: RecoveryResult | None = None

                if self.recovery_engine is not None:
                    if not self.runtime.check_timeout():
                        return self._timeout_failure(
                            results, failure_decision=failure_decision
                        )
                    try:
                        recovery_result = self.recovery_engine.evaluate(
                            decision=failure_decision,
                            judge=recovery_check,
                        )
                    except Exception as recovery_exc:
                        reason = (
                            f"{reason}; recovery evaluation failed: "
                            f"{type(recovery_exc).__name__}"
                        )

                    if not self.runtime.check_timeout():
                        return self._timeout_failure(
                            results,
                            failure_decision=failure_decision,
                            recovery_result=recovery_result,
                        )

                    if (
                        recovery_result is not None
                        and recovery_result.status == RecoveryStatus.JUDGE_PASSED
                    ):
                        verified = (
                            recovery_result.observation is not None
                            and recovery_result.judge_result is not None
                            and recovery_result.judge_result.success
                        )
                        if verified:
                            try:
                                self.runtime.update_observation(
                                    recovery_result.observation
                                )
                            except Exception as state_exc:
                                reason = (
                                    f"{reason}; recovery state update failed: "
                                    f"{type(state_exc).__name__}"
                                )
                            else:
                                if not self.runtime.check_timeout():
                                    return self._timeout_failure(
                                        results,
                                        failure_decision=failure_decision,
                                        recovery_result=recovery_result,
                                    )
                                self.runtime.mark_done()
                                return AgentLoopResult(
                                    success=True,
                                    transactions=results,
                                    judge_result=recovery_result.judge_result,
                                    failure_decision=failure_decision,
                                    recovery_result=recovery_result,
                                )
                        else:
                            reason = f"{reason}; incomplete recovery verification"

                # Only a verified pre-dispatch Grounding miss may trigger
                # replanning; an execution-stage error is never replayed.
                if (
                    recovery_result is not None
                    and recovery_result.status == RecoveryStatus.NEEDS_REPLAN
                    and self.replanner is not None
                    and failure_decision.phase == FailurePhase.GROUNDING
                    and not failure_decision.action_may_have_executed
                ):
                    if replan_count >= self.replanner.max_replans:
                        reason = f"{reason}; replan limit exceeded"
                    else:
                        available_steps = (
                            self.runtime.state.max_steps
                            - self.runtime.state.step_count
                        )
                        if available_steps <= 0:
                            reason = f"{reason}; no remaining step budget"
                        elif not self.runtime.check_timeout():
                            return self._timeout_failure(
                                results,
                                failure_decision=failure_decision,
                                recovery_result=recovery_result,
                            )
                        else:
                            try:
                                # Only use a fresh post-failure screen, and
                                # count observation/planner latency in timeout.
                                observation = (
                                    self.transaction.environment.observe(
                                        use_ocr=False
                                    )
                                )
                                self.runtime.update_observation(observation)

                                if not self.runtime.check_timeout():
                                    return self._timeout_failure(
                                        results,
                                        failure_decision=failure_decision,
                                        recovery_result=recovery_result,
                                    )

                                context = ReplanContext(
                                    task=self.runtime.state.task,
                                    completed_steps=tuple(completed_steps),
                                    failed_step=step,
                                    pending_steps=tuple(pending_steps),
                                    failure_reason=reason,
                                    attempt=replan_count + 1,
                                    available_steps=available_steps,
                                )
                                with temporary_replan_screenshot(
                                    observation
                                ) as screenshot_path:
                                    new_plan = self.replanner.replan(
                                        context,
                                        screenshot_path=screenshot_path,
                                    )
                                    logger.info(
                                        "replan generated",
                                        extra={
                                            "actions": [
                                                step.action.action_type.value
                                                for step in new_plan.steps
                                                if step.action is not None
                                            ]
                                        }
                                    )

                                if not self.runtime.check_timeout():
                                    return self._timeout_failure(
                                        results,
                                        failure_decision=failure_decision,
                                        recovery_result=recovery_result,
                                    )

                                # Plan step IDs restart at 1, but AgentRuntime
                                # step_count and timeout are never reset.
                                actions = [
                                    step.action
                                    for step in new_plan.steps
                                    if step.action is not None
                                ]

                                decision = self.action_policy.validate_plan(
                                    actions
                                )

                                if not decision.accepted:
                                    raise RuntimeError(
                                        f"action policy rejected: "
                                        f"{decision.reason}"
                                    )

                                new_steps = []

                                for idx, action in enumerate(
                                        decision.actions,
                                        start=1,
                                ):
                                    new_steps.append(
                                        PlanStep(
                                            step_id=idx,
                                            description=(
                                                f"replanned action {idx}"
                                            ),
                                            action=action,
                                        )
                                    )

                                pending_steps = new_steps

                                replan_count += 1

                                continue
                            except Exception as replan_exc:
                                if not self.runtime.check_timeout():
                                    return self._timeout_failure(
                                        results,
                                        failure_decision=failure_decision,
                                        recovery_result=recovery_result,
                                    )
                                reason = (
                                    f"{reason}; replanning failed: "
                                    f"{type(replan_exc).__name__}: {replan_exc}"
                                )

                self.runtime.mark_failed(reason)
                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=reason,
                    failure_decision=failure_decision,
                    recovery_result=recovery_result,
                )

        if not self.runtime.check_timeout():
            return self._timeout_failure(results)

        judge_result: SuccessJudgeResult | None = None
        if final_check is not None:
            try:
                judge_result = final_check()
                if not isinstance(judge_result, SuccessJudgeResult):
                    raise TypeError(
                        "final_check must return SuccessJudgeResult"
                    )
            except Exception as exc:
                reason = f"success judge failed: {exc}"
                decision = decide_recovery(
                    exc, phase=FailurePhase.SUCCESS_JUDGE
                )
                self.runtime.mark_failed(reason)
                return AgentLoopResult(
                    success=False,
                    transactions=results,
                    failure_reason=reason,
                    failure_decision=decision,
                )

            if not self.runtime.check_timeout():
                return self._timeout_failure(results, judge_result=judge_result)

            if not judge_result.success:
                self.runtime.mark_failed(judge_result.reason)
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
