from dataclasses import dataclass

import pytest

from gui_agent.agent.grounding import (
    GroundingBridge,
    GroundingError,
)
from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
    Point,
)


@dataclass
class FakeOCRResult:
    text: str
    confidence: float
    bbox: tuple


def make_result(
    text="Firefox",
    confidence=0.95,
    bbox=((100, 100), (200, 100), (200, 140), (100, 140)),
):
    return FakeOCRResult(
        text=text,
        confidence=confidence,
        bbox=bbox,
    )


def test_ground_click():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    grounded = bridge.ground(
        action,
        ocr_results=[make_result()],
        image_size=(2560, 1440),
        region=(0, 0, 2560, 1440),
    )

    assert grounded.position == Point(150, 120)
    assert grounded.element == "Firefox"


def test_ground_with_region_mapping():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    grounded = bridge.ground(
        action,
        ocr_results=[
            make_result(
                bbox=(
                    (100, 100),
                    (200, 100),
                    (200, 200),
                    (100, 200),
                )
            )
        ],
        image_size=(1000, 500),
        region=(500, 200, 2000, 1000),
    )

    assert grounded.position == Point(800, 500)


def test_existing_position_is_preserved():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        position=Point(100, 200),
        element="Firefox",
    )

    grounded = bridge.ground(
        action,
        ocr_results=[],
        image_size=(2560, 1440),
        region=(0, 0, 2560, 1440),
    )

    assert grounded is action


def test_non_groundable_action_is_preserved():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.TYPE_TEXT,
        text="hello",
    )

    grounded = bridge.ground(
        action,
        ocr_results=[],
        image_size=(2560, 1440),
        region=(0, 0, 2560, 1440),
    )

    assert grounded is action


def test_missing_element_rejected():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
    )

    with pytest.raises(GroundingError):
        bridge.ground(
            action,
            ocr_results=[],
            image_size=(2560, 1440),
            region=(0, 0, 2560, 1440),
        )


def test_target_not_found():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    with pytest.raises(GroundingError):
        bridge.ground(
            action,
            ocr_results=[
                make_result(text="Chrome")
            ],
            image_size=(2560, 1440),
            region=(0, 0, 2560, 1440),
        )


def test_ambiguous_target():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    with pytest.raises(GroundingError):
        bridge.ground(
            action,
            ocr_results=[
                make_result(),
                make_result(),
            ],
            image_size=(2560, 1440),
            region=(0, 0, 2560, 1440),
        )