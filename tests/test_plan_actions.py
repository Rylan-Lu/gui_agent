import pytest

from gui_agent.agent.action_adapter import (
    ActionValidationError,
    plan_step_to_action,
)
from gui_agent.agent.plan_schema import PlanStep, TaskPlan
from gui_agent.datasets.schema import ActionType


def test_parse_plan_with_click_action():
    plan = TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": "Click Firefox",
                    "action": {
                        "action_type": "click",
                        "element": "Firefox",
                    },
                }
            ]
        }
    )

    action = plan.steps[0].action

    assert action is not None
    assert action.action_type == ActionType.CLICK
    assert action.element == "Firefox"


def test_parse_keyboard_action():
    plan = TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": "Press Enter",
                    "action": {
                        "action_type": "key_press",
                        "keys": ["enter"],
                    },
                }
            ]
        }
    )

    assert plan.steps[0].action.keys == ("enter",)


def test_legacy_plan_still_supported():
    plan = TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": "Open browser",
                }
            ]
        }
    )

    assert plan.steps[0].action is None


def test_invalid_action_type():
    with pytest.raises(ValueError):
        TaskPlan.from_dict(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Invalid",
                        "action": {
                            "action_type": "invalid",
                        },
                    }
                ]
            }
        )


def test_plan_step_to_action():
    plan = TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": "Type hello",
                    "action": {
                        "action_type": "type_text",
                        "text": "hello",
                    },
                }
            ]
        }
    )

    action = plan_step_to_action(plan.steps[0])

    assert action.action_type == ActionType.TYPE_TEXT
    assert action.text == "hello"


def test_plan_step_without_action_rejected():
    step = PlanStep(
        step_id=1,
        description="Legacy step",
    )

    with pytest.raises(ActionValidationError):
        plan_step_to_action(step)