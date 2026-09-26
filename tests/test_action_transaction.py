import numpy as np

from gui_agent.agent.action_runner import ActionRunner
from gui_agent.agent.action_transaction import ActionTransaction
from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.observation import Observation
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
)
from gui_agent.ocr.base import OCRResult


class FakeController:
    def __init__(self):
        self.calls = []

    def click(self, point, *, button="left"):
        self.calls.append(
            ("click", point, button)
        )

    def type_text(self, text):
        self.calls.append(
            ("type_text", text)
        )


class FakeEnvironment:
    def __init__(self):
        self.calls = []
        self.index = 0

    def observe(self, *, use_ocr=False):
        self.calls.append(use_ocr)

        image = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        if self.index > 0:
            image[10:20, 10:20] = 255

        self.index += 1

        ocr_result = None

        if use_ocr:
            ocr_result = [
                OCRResult(
                    text="Firefox",
                    confidence=0.98,
                    bbox=(
                        (10, 10),
                        (30, 10),
                        (30, 30),
                        (10, 30),
                    ),
                )
            ]

        return Observation(
            screenshot=image,
            screen_width=100,
            screen_height=100,
            ocr_result=ocr_result,
            metadata={
                "image_size": (100, 100),
                "region": (0, 0, 100, 100),
            },
        )


def test_click_transaction():
    controller = FakeController()
    environment = FakeEnvironment()

    transaction = ActionTransaction(
        environment,
        ActionRunner(
            ActionExecutor(controller)
        ),
        settle_seconds=0,
    )

    result = transaction.execute(
        GUIAction(
            action_type=ActionType.CLICK,
            element="Firefox",
        )
    )

    assert environment.calls == [
        True,
        False,
    ]

    assert controller.calls == [
        ("click", (20, 20), "left")
    ]

    assert result.feedback.changed is True
    assert result.action.position is not None


def test_type_text_transaction_without_ocr():
    controller = FakeController()
    environment = FakeEnvironment()

    transaction = ActionTransaction(
        environment,
        ActionRunner(
            ActionExecutor(controller)
        ),
        settle_seconds=0,
    )

    result = transaction.execute(
        GUIAction(
            action_type=ActionType.TYPE_TEXT,
            text="hello",
        )
    )

    assert environment.calls == [
        False,
        False,
    ]

    assert controller.calls == [
        ("type_text", "hello")
    ]

    assert result.feedback.changed is True