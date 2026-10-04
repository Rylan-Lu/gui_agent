import pytest

from gui_agent.agent.action_adapter import (
    ActionValidationError,
    is_executable_action,
    validate_executable_action,
)
from gui_agent.datasets.schema import (
    ActionType,
    CoordinateSpace,
    GUIAction,
    Point,
)


@pytest.mark.parametrize(
    "action",
    [
        GUIAction(
            action_type=(
                ActionType.CLICK
            ),
            position=Point(
                100,
                200,
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.DOUBLE_CLICK
            ),
            position=Point(
                100,
                200,
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.MOVE
            ),
            position=Point(
                100,
                200,
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.DRAG
            ),
            position=Point(
                100,
                200,
            ),
            end_position=Point(
                300,
                400,
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.SCROLL
            ),
            scroll_delta=-3,
        ),
        GUIAction(
            action_type=(
                ActionType.TYPE_TEXT
            ),
            text="hello",
        ),
        GUIAction(
            action_type=(
                ActionType.KEY_PRESS
            ),
            keys=("enter",),
        ),
        GUIAction(
            action_type=(
                ActionType.HOTKEY
            ),
            keys=(
                "ctrl",
                "l",
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.WAIT
            ),
            wait_seconds=1.0,
        ),
    ],
)
def test_valid_executable_actions(
    action,
):
    assert is_executable_action(
        action
    )

    assert (
        validate_executable_action(
            action
        )
        is action
    )


@pytest.mark.parametrize(
    "action_type",
    [
        ActionType.PLAN,
        ActionType.EVALUATE,
        ActionType.SELECT,
        ActionType.MOUSE_DOWN,
        ActionType.MOUSE_UP,
        ActionType.OTHER,
    ],
)
def test_non_executable_actions(
    action_type,
):
    action = GUIAction(
        action_type=action_type
    )

    assert (
        is_executable_action(
            action
        )
        is False
    )

    with pytest.raises(
        ActionValidationError
    ):
        validate_executable_action(
            action
        )


def test_click_requires_position():
    action = GUIAction(
        action_type=(
            ActionType.CLICK
        )
    )

    with pytest.raises(
        ActionValidationError,
        match="position",
    ):
        validate_executable_action(
            action
        )


def test_drag_requires_end_position():
    action = GUIAction(
        action_type=(
            ActionType.DRAG
        ),
        position=Point(
            100,
            200,
        ),
    )

    with pytest.raises(
        ActionValidationError,
        match="end_position",
    ):
        validate_executable_action(
            action
        )


@pytest.mark.parametrize(
    "point",
    [
        Point(
            "100",
            200,
        ),
        Point(
            100,
            "200",
        ),
        Point(
            True,
            200,
        ),
        Point(
            100,
            float("inf"),
        ),
        Point(
            float("nan"),
            200,
        ),
    ],
)
def test_positional_actions_require_finite_numeric_points(
    point,
):
    action = GUIAction(
        action_type=(
            ActionType.CLICK
        ),
        position=point,
    )

    with pytest.raises(
        ActionValidationError
    ):
        validate_executable_action(
            action
        )


def test_scroll_requires_delta():
    action = GUIAction(
        action_type=(
            ActionType.SCROLL
        )
    )

    with pytest.raises(
        ActionValidationError,
        match="scroll_delta",
    ):
        validate_executable_action(
            action
        )


@pytest.mark.parametrize(
    "delta",
    [
        1.5,
        "3",
        True,
    ],
)
def test_scroll_requires_integer_delta(
    delta,
):
    action = GUIAction(
        action_type=(
            ActionType.SCROLL
        ),
        scroll_delta=delta,
    )

    with pytest.raises(
        ActionValidationError,
        match="scroll_delta",
    ):
        validate_executable_action(
            action
        )


def test_type_text_requires_text():
    action = GUIAction(
        action_type=(
            ActionType.TYPE_TEXT
        )
    )

    with pytest.raises(
        ActionValidationError,
        match="string text",
    ):
        validate_executable_action(
            action
        )


def test_type_text_allows_empty_string():
    action = GUIAction(
        action_type=(
            ActionType.TYPE_TEXT
        ),
        text="",
    )

    assert (
        validate_executable_action(
            action
        )
        is action
    )


@pytest.mark.parametrize(
    "keys",
    [
        (),
        (
            "ctrl",
            "c",
        ),
        ("",),
        ("   ",),
        (123,),
    ],
)
def test_key_press_requires_one_non_empty_string(
    keys,
):
    action = GUIAction(
        action_type=(
            ActionType.KEY_PRESS
        ),
        keys=keys,
    )

    with pytest.raises(
        ActionValidationError
    ):
        validate_executable_action(
            action
        )


@pytest.mark.parametrize(
    "keys",
    [
        (),
        (
            "ctrl",
            "",
        ),
        (
            "ctrl",
            123,
        ),
    ],
)
def test_hotkey_requires_non_empty_string_keys(
    keys,
):
    action = GUIAction(
        action_type=(
            ActionType.HOTKEY
        ),
        keys=keys,
    )

    with pytest.raises(
        ActionValidationError
    ):
        validate_executable_action(
            action
        )


@pytest.mark.parametrize(
    "wait_seconds",
    [
        -1,
        "1",
        True,
        float("inf"),
        float("nan"),
    ],
)
def test_wait_rejects_invalid_seconds(
    wait_seconds,
):
    action = GUIAction(
        action_type=(
            ActionType.WAIT
        ),
        wait_seconds=(
            wait_seconds
        ),
    )

    with pytest.raises(
        ActionValidationError
    ):
        validate_executable_action(
            action
        )


def test_normalized_coordinates_rejected_for_positional_action():
    action = GUIAction(
        action_type=(
            ActionType.CLICK
        ),
        position=Point(
            0.5,
            0.5,
        ),
        coordinate_space=(
            CoordinateSpace.NORMALIZED
        ),
    )

    with pytest.raises(
        ActionValidationError,
        match="pixel coordinates",
    ):
        validate_executable_action(
            action
        )


@pytest.mark.parametrize(
    "action",
    [
        GUIAction(
            action_type=(
                ActionType.TYPE_TEXT
            ),
            text="hello",
            coordinate_space=(
                CoordinateSpace.NORMALIZED
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.KEY_PRESS
            ),
            keys=("enter",),
            coordinate_space=(
                CoordinateSpace.NORMALIZED
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.HOTKEY
            ),
            keys=(
                "ctrl",
                "l",
            ),
            coordinate_space=(
                CoordinateSpace.NORMALIZED
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.SCROLL
            ),
            scroll_delta=-3,
            coordinate_space=(
                CoordinateSpace.NORMALIZED
            ),
        ),
        GUIAction(
            action_type=(
                ActionType.WAIT
            ),
            wait_seconds=1.0,
            coordinate_space=(
                CoordinateSpace.NORMALIZED
            ),
        ),
    ],
)
def test_coordinate_free_actions_ignore_coordinate_space(
    action,
):
    assert (
        validate_executable_action(
            action
        )
        is action
    )


def test_invalid_mouse_button():
    action = GUIAction(
        action_type=(
            ActionType.CLICK
        ),
        position=Point(
            100,
            200,
        ),
        mouse_button="invalid",
    )

    with pytest.raises(
        ActionValidationError,
        match="mouse button",
    ):
        validate_executable_action(
            action
        )