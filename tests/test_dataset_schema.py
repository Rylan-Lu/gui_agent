import pytest

from gui_agent.datasets.schema import (
    ActionType,
    CoordinateSpace,
    GUIAction,
    GUIExample,
    Point,
)


def test_point_creation():
    point = Point(x=100, y=200)

    assert point.x == 100
    assert point.y == 200


def test_click_action():
    action = GUIAction(
        action_type=ActionType.CLICK,
        position=Point(100, 200),
    )

    assert action.action_type == ActionType.CLICK
    assert action.position == Point(100, 200)


def test_type_text_action():
    action = GUIAction(
        action_type=ActionType.TYPE_TEXT,
        text="GUI Agent",
    )

    assert action.text == "GUI Agent"


def test_normalized_coordinate_space():
    action = GUIAction(
        action_type=ActionType.CLICK,
        position=Point(0.5, 0.25),
        coordinate_space=CoordinateSpace.NORMALIZED,
    )

    assert action.coordinate_space == CoordinateSpace.NORMALIZED


def test_gui_example_creation():
    action = GUIAction(
        action_type=ActionType.CLICK,
        position=Point(500, 300),
    )

    example = GUIExample(
        source="screenagent",
        task_id="task_001",
        instruction="Click the search box",
        screenshot="0001.png",
        action=action,
    )

    assert example.source == "screenagent"
    assert example.task_id == "task_001"
    assert example.step_index == 0
    assert example.action == action


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", ""),
        ("task_id", ""),
        ("instruction", ""),
    ],
)
def test_required_text_fields_must_not_be_empty(field, value):
    kwargs = {
        "source": "screenagent",
        "task_id": "task_001",
        "instruction": "Click something",
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        GUIExample(**kwargs)


def test_negative_step_index_rejected():
    with pytest.raises(ValueError):
        GUIExample(
            source="screenagent",
            task_id="task_001",
            instruction="Click something",
            step_index=-1,
        )


def test_history():
    first_action = GUIAction(
        action_type=ActionType.CLICK,
        position=Point(100, 200),
    )

    second_action = GUIAction(
        action_type=ActionType.TYPE_TEXT,
        text="hello",
    )

    example = GUIExample(
        source="screenagent",
        task_id="task_001",
        instruction="Search for hello",
        step_index=1,
        action=second_action,
        history=(first_action,),
    )

    assert len(example.history) == 1
    assert example.history[0] == first_action