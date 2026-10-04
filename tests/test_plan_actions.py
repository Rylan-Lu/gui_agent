import pytest

from gui_agent.agent.action_adapter import (
    ActionValidationError,
    plan_step_to_action,
)
from gui_agent.agent.plan_schema import (
    PlanStep,
    TaskPlan,
)
from gui_agent.datasets.schema import (
    ActionType,
)


def make_plan_with_action(
    action: dict,
):
    return TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": (
                        "Test action"
                    ),
                    "action": action,
                }
            ]
        }
    )


def test_parse_plan_with_click_action():
    plan = make_plan_with_action(
        {
            "action_type": (
                "click"
            ),
            "element": "Firefox",
        }
    )

    action = (
        plan.steps[0].action
    )

    assert action is not None

    assert (
        action.action_type
        == ActionType.CLICK
    )

    assert (
        action.element
        == "Firefox"
    )


def test_parse_keyboard_action():
    plan = make_plan_with_action(
        {
            "action_type": (
                "key_press"
            ),
            "keys": [
                "enter"
            ],
        }
    )

    assert (
        plan.steps[0]
        .action
        .keys
        == ("enter",)
    )


def test_legacy_plan_still_supported():
    plan = TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": (
                        "Open browser"
                    ),
                }
            ]
        }
    )

    assert (
        plan.steps[0].action
        is None
    )


def test_invalid_action_type():
    with pytest.raises(
        ValueError,
        match="invalid action_type",
    ):
        make_plan_with_action(
            {
                "action_type": (
                    "invalid"
                ),
            }
        )


@pytest.mark.parametrize(
    "action_type",
    [
        "plan",
        "evaluate",
        "move",
        "drag",
        "mouse_down",
        "mouse_up",
        "select",
        "other",
    ],
)
def test_planner_rejects_action_types_outside_contract(
    action_type,
):
    with pytest.raises(
        ValueError,
        match=(
            "not allowed in "
            "planner output"
        ),
    ):
        make_plan_with_action(
            {
                "action_type": (
                    action_type
                ),
            }
        )


def test_plan_step_to_action():
    plan = make_plan_with_action(
        {
            "action_type": (
                "type_text"
            ),
            "text": "hello",
        }
    )

    action = (
        plan_step_to_action(
            plan.steps[0]
        )
    )

    assert (
        action.action_type
        == ActionType.TYPE_TEXT
    )

    assert (
        action.text
        == "hello"
    )


def test_plan_step_without_action_rejected():
    step = PlanStep(
        step_id=1,
        description="Legacy step",
    )

    with pytest.raises(
        ActionValidationError
    ):
        plan_step_to_action(
            step
        )


@pytest.mark.parametrize(
    "action",
    [
        {
            "action_type": "click",
            "element": "   ",
        },
        {
            "action_type": (
                "double_click"
            ),
            "element": None,
        },
        {
            "action_type": (
                "type_text"
            ),
            "text": 123,
        },
        {
            "action_type": (
                "key_press"
            ),
            "keys": [],
        },
        {
            "action_type": (
                "key_press"
            ),
            "keys": [
                "ctrl",
                "c",
            ],
        },
        {
            "action_type": (
                "key_press"
            ),
            "keys": [
                123
            ],
        },
        {
            "action_type": (
                "hotkey"
            ),
            "keys": [],
        },
        {
            "action_type": (
                "hotkey"
            ),
            "keys": [
                "ctrl",
                "",
            ],
        },
        {
            "action_type": (
                "scroll"
            ),
            "scroll_delta": 1.5,
        },
        {
            "action_type": (
                "scroll"
            ),
            "scroll_delta": True,
        },
        {
            "action_type": "wait",
            "wait_seconds": -1,
        },
        {
            "action_type": "wait",
            "wait_seconds": "1",
        },
        {
            "action_type": "wait",
            "wait_seconds": (
                float("inf")
            ),
        },
    ],
)
def test_planner_rejects_invalid_action_payload(
    action,
):
    with pytest.raises(
        ValueError
    ):
        make_plan_with_action(
            action
        )


def test_planner_rejects_unknown_action_field():
    with pytest.raises(
        ValueError,
        match="unknown field",
    ):
        make_plan_with_action(
            {
                "action_type": (
                    "wait"
                ),
                "wait_second": 2,
            }
        )


def test_planner_rejects_extra_field_for_action_type():
    with pytest.raises(
        ValueError,
        match="unknown field",
    ):
        make_plan_with_action(
            {
                "action_type": (
                    "type_text"
                ),
                "text": "hello",
                "element": "Search",
            }
        )