import pytest

import gui_agent.agent.executor as executor_module

from gui_agent.agent.action_adapter import (
    ActionValidationError,
)
from gui_agent.agent.executor import (
    ActionExecutionError,
    ActionExecutor,
)
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
    Point,
)


class FakeController:
    def __init__(
        self,
    ):
        self.calls = []

    def click(
        self,
        point,
        *,
        button="left",
    ):
        self.calls.append(
            (
                "click",
                point,
                button,
            )
        )

    def double_click(
        self,
        point,
        *,
        button="left",
    ):
        self.calls.append(
            (
                "double_click",
                point,
                button,
            )
        )

    def move_to(
        self,
        point,
    ):
        self.calls.append(
            (
                "move_to",
                point,
            )
        )

    def drag_to(
        self,
        point,
        *,
        button="left",
    ):
        self.calls.append(
            (
                "drag_to",
                point,
                button,
            )
        )

    def scroll(
        self,
        amount,
    ):
        self.calls.append(
            (
                "scroll",
                amount,
            )
        )

    def type_text(
        self,
        text,
    ):
        self.calls.append(
            (
                "type_text",
                text,
            )
        )

    def press(
        self,
        key,
    ):
        self.calls.append(
            (
                "press",
                key,
            )
        )

    def hotkey(
        self,
        *keys,
    ):
        self.calls.append(
            (
                "hotkey",
                keys,
            )
        )


class FailingController(
    FakeController
):
    def click(
        self,
        point,
        *,
        button="left",
    ):
        raise RuntimeError(
            "controller failed"
        )


def test_execute_click():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.CLICK
            ),
            position=Point(
                100,
                200,
            ),
        )
    )

    assert (
        controller.calls
        == [
            (
                "click",
                (
                    100,
                    200,
                ),
                "left",
            )
        ]
    )


def test_execute_click_preserves_mouse_button():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.CLICK
            ),
            position=Point(
                100,
                200,
            ),
            mouse_button="right",
        )
    )

    assert (
        controller.calls
        == [
            (
                "click",
                (
                    100,
                    200,
                ),
                "right",
            )
        ]
    )


def test_execute_double_click():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.DOUBLE_CLICK
            ),
            position=Point(
                10,
                20,
            ),
        )
    )

    assert (
        controller.calls
        == [
            (
                "double_click",
                (
                    10,
                    20,
                ),
                "left",
            )
        ]
    )


def test_execute_move():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.MOVE
            ),
            position=Point(
                50,
                60,
            ),
        )
    )

    assert (
        controller.calls
        == [
            (
                "move_to",
                (
                    50,
                    60,
                ),
            )
        ]
    )


def test_execute_drag():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.DRAG
            ),
            position=Point(
                10,
                20,
            ),
            end_position=Point(
                100,
                200,
            ),
        )
    )

    assert (
        controller.calls
        == [
            (
                "move_to",
                (
                    10,
                    20,
                ),
            ),
            (
                "drag_to",
                (
                    100,
                    200,
                ),
                "left",
            ),
        ]
    )


def test_execute_scroll():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.SCROLL
            ),
            scroll_delta=-3,
        )
    )

    assert (
        controller.calls
        == [
            (
                "scroll",
                -3,
            )
        ]
    )


def test_execute_type_text():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.TYPE_TEXT
            ),
            text="hello",
        )
    )

    assert (
        controller.calls
        == [
            (
                "type_text",
                "hello",
            )
        ]
    )


def test_execute_key_press():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.KEY_PRESS
            ),
            keys=("enter",),
        )
    )

    assert (
        controller.calls
        == [
            (
                "press",
                "enter",
            )
        ]
    )


def test_execute_hotkey():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.HOTKEY
            ),
            keys=(
                "ctrl",
                "l",
            ),
        )
    )

    assert (
        controller.calls
        == [
            (
                "hotkey",
                (
                    "ctrl",
                    "l",
                ),
            )
        ]
    )


def test_execute_wait(
    monkeypatch,
):
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    sleep_calls = []

    monkeypatch.setattr(
        executor_module.time,
        "sleep",
        sleep_calls.append,
    )

    executor.execute(
        GUIAction(
            action_type=(
                ActionType.WAIT
            ),
            wait_seconds=1.25,
        )
    )

    assert (
        sleep_calls
        == [1.25]
    )

    assert (
        controller.calls
        == []
    )


def test_invalid_action_preserves_validation_error():
    controller = (
        FakeController()
    )

    executor = (
        ActionExecutor(
            controller
        )
    )

    with pytest.raises(
        ActionValidationError
    ):
        executor.execute(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
            )
        )


def test_controller_failure_is_wrapped_as_execution_error():
    executor = (
        ActionExecutor(
            FailingController()
        )
    )

    with pytest.raises(
        ActionExecutionError,
        match=(
            "failed to execute click"
        ),
    ) as exc_info:
        executor.execute(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                position=Point(
                    100,
                    200,
                ),
            )
        )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    assert (
        str(
            exc_info.value.__cause__
        )
        == "controller failed"
    )