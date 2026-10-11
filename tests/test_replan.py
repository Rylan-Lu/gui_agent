from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from gui_agent.agent.plan_schema import PlanStep, TaskPlan
from gui_agent.agent.replan import (
    ReplanContext,
    ReplanError,
    Replanner,
    build_replan_instruction,
    temporary_replan_screenshot,
)
from gui_agent.datasets.schema import ActionType, GUIAction


def make_context(*, attempt=1, available_steps=3):
    return ReplanContext(
        task="Open editor and type hello",
        completed_steps=(
            PlanStep(
                step_id=1,
                description="Open editor",
                action=GUIAction(
                    action_type=ActionType.HOTKEY,
                    keys=("ctrl", "n"),
                ),
            ),
        ),
        failed_step=PlanStep(
            step_id=2,
            description="Click editor",
            action=GUIAction(
                action_type=ActionType.CLICK,
                element="Editor",
            ),
        ),
        pending_steps=(
            PlanStep(
                step_id=3,
                description="Type hello",
                action=GUIAction(
                    action_type=ActionType.TYPE_TEXT,
                    text="hello",
                ),
            ),
        ),
        failure_reason="target not found",
        attempt=attempt,
        available_steps=available_steps,
    )


def valid_plan():
    return TaskPlan(
        steps=(
            PlanStep(
                step_id=1,
                description="Type hello",
                action=GUIAction(
                    action_type=ActionType.TYPE_TEXT,
                    text="hello",
                ),
            ),
        )
    )


def test_replan_prompt_contains_execution_context():
    prompt = build_replan_instruction(make_context())
    assert "Open editor and type hello" in prompt
    assert "target not found" in prompt
    assert "Click editor" in prompt
    assert "completed_steps" in prompt
    assert "available_steps" in prompt


def test_replanner_calls_existing_planner():
    planner = Mock()
    planner.plan.return_value = valid_plan()
    replanner = Replanner(planner)
    screenshot = Path("current_screen.png")
    result = replanner.replan(make_context(), screenshot_path=screenshot)
    assert result == valid_plan()
    planner.plan.assert_called_once()
    assert planner.plan.call_args.kwargs["screenshot_path"] == screenshot


def test_replanner_rejects_non_executable_plan():
    planner = Mock()
    planner.plan.return_value = TaskPlan(
        steps=(PlanStep(step_id=1, description="Think about next action"),)
    )
    with pytest.raises(ReplanError, match="non-executable"):
        Replanner(planner).replan(make_context())


def test_replanner_rejects_excessive_steps():
    planner = Mock()
    planner.plan.return_value = TaskPlan(
        steps=(
            PlanStep(
                step_id=1,
                description="Type hello",
                action=GUIAction(action_type=ActionType.TYPE_TEXT, text="hello"),
            ),
            PlanStep(
                step_id=2,
                description="Press Enter",
                action=GUIAction(action_type=ActionType.KEY_PRESS, keys=("enter",)),
            ),
        )
    )
    with pytest.raises(ReplanError, match="step budget"):
        Replanner(planner).replan(make_context(available_steps=1))


def test_replanner_enforces_replan_limit():
    planner = Mock()
    replanner = Replanner(planner, max_replans=1)
    with pytest.raises(ReplanError, match="attempt limit"):
        replanner.replan(make_context(attempt=2))
    planner.plan.assert_not_called()


def test_replanner_rejects_invalid_output():
    planner = Mock()
    planner.plan.return_value = {"steps": []}
    with pytest.raises(ReplanError, match="TaskPlan"):
        Replanner(planner).replan(make_context())


def test_context_rejects_invalid_step_budget():
    with pytest.raises(ValueError, match="available_steps"):
        make_context(available_steps=0)


def test_invalid_max_replans_rejected():
    with pytest.raises(ValueError):
        Replanner(Mock(), max_replans=-1)


def test_replanner_rejects_repeated_failed_action():
    context = make_context()
    planner = Mock()
    planner.plan.return_value = TaskPlan(
        steps=(
            PlanStep(
                step_id=1,
                description="Click editor again",
                action=context.failed_step.action,
            ),
        )
    )
    with pytest.raises(ReplanError, match="repeats the failed action"):
        Replanner(planner).replan(context)


def test_replanner_rejects_repeated_completed_action():
    context = make_context()
    planner = Mock()
    planner.plan.return_value = TaskPlan(
        steps=(
            PlanStep(
                step_id=1,
                description="Open editor again",
                action=context.completed_steps[0].action,
            ),
        )
    )
    with pytest.raises(ReplanError, match="repeats a completed action"):
        Replanner(planner).replan(context)


def test_temporary_screenshot_converts_bgr_and_is_cleaned():
    # BGR (255, 0, 0) should appear as RGB (0, 0, 255) = blue.
    observation = Mock()
    observation.screenshot = np.array([[[255, 0, 0]]], dtype=np.uint8)
    with temporary_replan_screenshot(observation) as path:
        assert path.is_file()
        with Image.open(path) as image:
            assert image.convert("RGB").getpixel((0, 0)) == (0, 0, 255)
    assert not path.exists()


def test_temporary_screenshot_removed_on_exception():
    observation = Mock()
    observation.screenshot = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        with temporary_replan_screenshot(observation) as path:
            raise RuntimeError("planner failed")
    assert not path.exists()


def test_invalid_screenshot_rejected():
    observation = Mock()
    observation.screenshot = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ReplanError, match="invalid recovery screenshot"):
        with temporary_replan_screenshot(observation):
            pass
