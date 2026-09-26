import numpy as np
import pytest

from gui_agent.agent.action_runner import (
    ActionRunner,
    ActionRunnerError,
)
from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.observation import Observation
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
    Point,
)
from gui_agent.ocr.base import OCRResult


class FakeController:
    def __init__(self):
        self.calls = []

    def click(self, point, *, button="left"):
        self.calls.append(("click", point, button))

    def type_text(self, text):
        self.calls.append(("type_text", text))


def make_observation():
    return Observation(
        screenshot=np.zeros(
            (1440, 2560, 3),
            dtype=np.uint8,
        ),
        screen_width=2560,
        screen_height=1440,
        ocr_result=[
            OCRResult(
                text="Firefox",
                confidence=0.98,
                bbox=(
                    (100, 100),
                    (200, 100),
                    (200, 140),
                    (100, 140),
                ),
            )
        ],
        metadata={
            "image_size": (2560, 1440),
            "region": (0, 0, 2560, 1440),
        },
    )


def test_ground_and_execute_click():
    controller = FakeController()
    runner = ActionRunner(
        ActionExecutor(controller)
    )

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    resolved = runner.run(
        action,
        make_observation(),
    )

    assert resolved.position == Point(150, 120)

    assert controller.calls == [
        ("click", (150, 120), "left")
    ]


def test_execute_action_without_grounding():
    controller = FakeController()
    runner = ActionRunner(
        ActionExecutor(controller)
    )

    action = GUIAction(
        action_type=ActionType.TYPE_TEXT,
        text="hello",
    )

    resolved = runner.run(
        action,
        make_observation(),
    )

    assert resolved is action
    assert controller.calls == [
        ("type_text", "hello")
    ]


def test_click_requires_ocr():
    controller = FakeController()
    runner = ActionRunner(
        ActionExecutor(controller)
    )

    observation = make_observation()
    observation.ocr_result = None

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    with pytest.raises(ActionRunnerError):
        runner.run(action, observation)


def test_existing_position_skips_grounding():
    controller = FakeController()
    runner = ActionRunner(
        ActionExecutor(controller)
    )

    action = GUIAction(
        action_type=ActionType.CLICK,
        position=Point(500, 600),
    )

    resolved = runner.run(
        action,
        make_observation(),
    )

    assert resolved.position == Point(500, 600)

    assert controller.calls == [
        ("click", (500, 600), "left")
    ]