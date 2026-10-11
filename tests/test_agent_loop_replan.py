from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from gui_agent.agent.failure_policy import FailurePhase, decide_recovery
from gui_agent.agent.grounding import GroundingTargetNotFoundError
from gui_agent.agent.loop import AgentLoop
from gui_agent.agent.plan_schema import PlanStep, TaskPlan
from gui_agent.agent.recovery import RecoveryEngine
from gui_agent.agent.replan import Replanner
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import SuccessJudgeResult
from gui_agent.datasets.schema import ActionType, GUIAction


def make_test_setup(
    monkeypatch,
    *,
    max_steps=4,
    failure_phase=FailurePhase.GROUNDING,
):
    monkeypatch.setattr(
        "gui_agent.agent.loop.plan_step_to_action",
        lambda step: step.action,
    )

    completed = PlanStep(
        step_id=1,
        description="Type first text",
        action=GUIAction(action_type=ActionType.TYPE_TEXT, text="first"),
    )
    failed = PlanStep(
        step_id=1,
        description="Click missing target",
        action=GUIAction(action_type=ActionType.CLICK, element="Missing"),
    )
    replacement = PlanStep(
        step_id=1,
        description="Type alternative text",
        action=GUIAction(action_type=ActionType.TYPE_TEXT, text="alternative"),
    )
    plan = TaskPlan(steps=(failed,))
    new_plan = TaskPlan(steps=(replacement,))

    observation = SimpleNamespace(
        screenshot=np.zeros((10, 12, 3), dtype=np.uint8)
    )
    environment = Mock()
    environment.observe.return_value = observation

    error = GroundingTargetNotFoundError("target missing")
    transaction = Mock()
    transaction.environment = environment

    def execute(action):
        if action.action_type == ActionType.CLICK:
            actual_error = (
                error
                if failure_phase == FailurePhase.GROUNDING
                else RuntimeError("execution uncertain")
            )
            transaction.last_failure = decide_recovery(
                actual_error, phase=failure_phase
            )
            raise actual_error
        transaction.last_failure = None
        return SimpleNamespace(after=observation)

    transaction.execute.side_effect = execute
    planner = Mock()
    planner.plan.return_value = new_plan
    replanner = Replanner(planner, max_replans=1)
    runtime = AgentRuntime.create(
        "complete test task", max_steps=max_steps, timeout_s=60
    )
    loop = AgentLoop(
        runtime=runtime,
        transaction=transaction,
        recovery_engine=RecoveryEngine(environment),
        replanner=replanner,
    )
    return SimpleNamespace(
        loop=loop,
        runtime=runtime,
        transaction=transaction,
        planner=planner,
        environment=environment,
        plan=plan,
        completed=completed,
        failed=failed,
        replacement=replacement,
    )


def test_replan_executes_new_plan(monkeypatch):
    setup = make_test_setup(monkeypatch)
    screenshot_paths = []

    def plan_from_screenshot(instruction, screenshot_path=None):
        assert screenshot_path.exists()
        screenshot_paths.append(screenshot_path)
        assert "target missing" in instruction
        return TaskPlan(steps=(setup.replacement,))

    setup.planner.plan.side_effect = plan_from_screenshot
    result = setup.loop.run(
        setup.plan,
        final_check=lambda: SuccessJudgeResult(success=True, reason="verified"),
    )

    assert result.success is True
    assert result.judge_result.success is True
    assert setup.runtime.state.done is True
    assert setup.runtime.state.step_count == 2
    assert setup.transaction.execute.call_count == 2
    assert setup.planner.plan.call_count == 1
    assert not screenshot_paths[0].exists()


def test_uncertain_execution_never_replans(monkeypatch):
    setup = make_test_setup(
        monkeypatch, failure_phase=FailurePhase.ACTION_EXECUTION
    )
    result = setup.loop.run(setup.plan)
    assert result.success is False
    setup.planner.plan.assert_not_called()
    setup.transaction.execute.assert_called_once()


def test_step_budget_prevents_replan(monkeypatch):
    setup = make_test_setup(monkeypatch, max_steps=1)
    result = setup.loop.run(setup.plan)
    assert result.success is False
    assert "step budget" in result.failure_reason
    setup.planner.plan.assert_not_called()
    setup.transaction.execute.assert_called_once()


def test_completed_action_not_replayed(monkeypatch):
    setup = make_test_setup(monkeypatch)
    original_plan = TaskPlan(
        steps=(
            setup.completed,
            PlanStep(
                step_id=2,
                description=setup.failed.description,
                action=setup.failed.action,
            ),
        )
    )
    setup.planner.plan.return_value = TaskPlan(
        steps=(
            PlanStep(
                step_id=1,
                description="Repeat the first text",
                action=setup.completed.action,
            ),
        )
    )
    result = setup.loop.run(original_plan)
    assert result.success is False
    assert "repeats a completed action" in result.failure_reason
    assert setup.transaction.execute.call_count == 2
    assert setup.runtime.state.step_count == 2


def test_failed_action_cannot_be_replayed(monkeypatch):
    setup = make_test_setup(monkeypatch)
    setup.planner.plan.return_value = TaskPlan(
        steps=(
            PlanStep(
                step_id=1,
                description="Try missing target again",
                action=setup.failed.action,
            ),
        )
    )
    result = setup.loop.run(setup.plan)
    assert result.success is False
    assert "repeats the failed action" in result.failure_reason
    setup.transaction.execute.assert_called_once()


def test_replan_limit_stops_second_grounding_failure(monkeypatch):
    setup = make_test_setup(monkeypatch, max_steps=5)
    another_missing_action = GUIAction(
        action_type=ActionType.CLICK, element="Also Missing"
    )
    setup.planner.plan.return_value = TaskPlan(
        steps=(
            PlanStep(
                step_id=1,
                description="Click a different missing target",
                action=another_missing_action,
            ),
        )
    )

    # Both clicks fail in this test. No second planner invocation is permitted.
    original_execute = setup.transaction.execute.side_effect

    def execute_any_click(action):
        if action.action_type == ActionType.CLICK:
            error = GroundingTargetNotFoundError("still missing")
            setup.transaction.last_failure = decide_recovery(
                error, phase=FailurePhase.GROUNDING
            )
            raise error
        return original_execute(action)

    setup.transaction.execute.side_effect = execute_any_click
    result = setup.loop.run(setup.plan)
    assert result.success is False
    assert "replan limit exceeded" in result.failure_reason
    assert setup.planner.plan.call_count == 1
    assert setup.transaction.execute.call_count == 2


def test_planner_timeout_cannot_report_success(monkeypatch):
    setup = make_test_setup(monkeypatch)

    def slow_planning(instruction, screenshot_path=None):
        setup.runtime.state.started_at -= 1000.0
        return TaskPlan(steps=(setup.replacement,))

    setup.planner.plan.side_effect = slow_planning
    result = setup.loop.run(setup.plan)
    assert result.success is False
    assert "timeout exceeded" in result.failure_reason
    setup.transaction.execute.assert_called_once()
