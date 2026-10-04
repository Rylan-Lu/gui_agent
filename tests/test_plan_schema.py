import pytest

from gui_agent.agent.plan_schema import (
    PlanStep,
    TaskPlan,
)


def test_valid_step():
    step = PlanStep(
        1,
        "Open browser",
    )

    assert step.step_id == 1
    assert (
        step.description
        == "Open browser"
    )


@pytest.mark.parametrize(
    "step_id",
    [
        0,
        -1,
        True,
        "1",
    ],
)
def test_invalid_step_id(
    step_id,
):
    with pytest.raises(
        ValueError
    ):
        PlanStep(
            step_id,
            "Open browser",
        )


def test_empty_description():
    with pytest.raises(
        ValueError
    ):
        PlanStep(
            1,
            "   ",
        )


def test_valid_plan():
    plan = TaskPlan(
        steps=(
            PlanStep(
                1,
                "Open browser",
            ),
            PlanStep(
                2,
                "Search GUI Agent",
            ),
        )
    )

    assert len(
        plan.steps
    ) == 2


def test_empty_plan():
    with pytest.raises(
        ValueError,
        match="between 1 and 8",
    ):
        TaskPlan(
            steps=()
        )


def test_plan_rejects_more_than_eight_steps():
    steps = tuple(
        PlanStep(
            index,
            f"Step {index}",
        )
        for index
        in range(
            1,
            10,
        )
    )

    with pytest.raises(
        ValueError,
        match="between 1 and 8",
    ):
        TaskPlan(
            steps=steps
        )


def test_invalid_step_order():
    with pytest.raises(
        ValueError
    ):
        TaskPlan(
            steps=(
                PlanStep(
                    1,
                    "First",
                ),
                PlanStep(
                    3,
                    "Third",
                ),
            )
        )


def test_parse_json():
    text = """
    {
        "steps": [
            {
                "step_id": 1,
                "description": "Open browser"
            }
        ]
    }
    """

    plan = (
        TaskPlan.from_json(
            text
        )
    )

    assert len(
        plan.steps
    ) == 1

    assert (
        plan.steps[0].description
        == "Open browser"
    )


def test_invalid_json():
    with pytest.raises(
        ValueError
    ):
        TaskPlan.from_json(
            "not json"
        )


def test_missing_steps():
    with pytest.raises(
        ValueError
    ):
        TaskPlan.from_dict(
            {}
        )


def test_invalid_step_object():
    with pytest.raises(
        ValueError
    ):
        TaskPlan.from_dict(
            {
                "steps": [
                    "Open browser"
                ]
            }
        )


def test_plan_rejects_unknown_top_level_field():
    with pytest.raises(
        ValueError,
        match="unknown field",
    ):
        TaskPlan.from_dict(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": (
                            "Open browser"
                        ),
                    }
                ],
                "reasoning": (
                    "extra model output"
                ),
            }
        )


def test_plan_step_rejects_unknown_field():
    with pytest.raises(
        ValueError,
        match="unknown field",
    ):
        TaskPlan.from_dict(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": (
                            "Open browser"
                        ),
                        "unexpected": True,
                    }
                ]
            }
        )