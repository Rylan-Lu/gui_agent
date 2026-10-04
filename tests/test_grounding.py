from dataclasses import dataclass

import pytest

from gui_agent.agent.grounding import (
    GroundingAmbiguousTargetError,
    GroundingBridge,
    GroundingError,
    GroundingTargetNotFoundError,
)
from gui_agent.datasets.schema import (
    ActionType,
    CoordinateSpace,
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
    bbox=(
        (100, 100),
        (200, 100),
        (200, 140),
        (100, 140),
    ),
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
        ocr_results=[
            make_result()
        ],
        image_size=(
            2560,
            1440,
        ),
        region=(
            0,
            0,
            2560,
            1440,
        ),
    )

    assert (
        grounded.position
        == Point(
            150,
            120,
        )
    )

    assert (
        grounded.element
        == "Firefox"
    )

    assert (
        grounded.coordinate_space
        == CoordinateSpace.PIXEL
    )


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
        image_size=(
            1000,
            500,
        ),
        region=(
            500,
            200,
            2000,
            1000,
        ),
    )

    assert (
        grounded.position
        == Point(
            800,
            500,
        )
    )

    assert (
        grounded.coordinate_space
        == CoordinateSpace.PIXEL
    )


def test_grounding_sets_pixel_coordinate_space():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
        coordinate_space=(
            CoordinateSpace.NORMALIZED
        ),
    )

    grounded = bridge.ground(
        action,
        ocr_results=[
            make_result()
        ],
        image_size=(
            2560,
            1440,
        ),
        region=(
            0,
            0,
            2560,
            1440,
        ),
    )

    assert (
        grounded.position
        == Point(
            150,
            120,
        )
    )

    assert (
        grounded.coordinate_space
        == CoordinateSpace.PIXEL
    )


def test_existing_position_is_preserved():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        position=Point(
            100,
            200,
        ),
        element="Firefox",
    )

    grounded = bridge.ground(
        action,
        ocr_results=[],
        image_size=(
            2560,
            1440,
        ),
        region=(
            0,
            0,
            2560,
            1440,
        ),
    )

    assert grounded is action


def test_non_groundable_action_is_preserved():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=(
            ActionType.TYPE_TEXT
        ),
        text="hello",
    )

    grounded = bridge.ground(
        action,
        ocr_results=[],
        image_size=(
            2560,
            1440,
        ),
        region=(
            0,
            0,
            2560,
            1440,
        ),
    )

    assert grounded is action


@pytest.mark.parametrize(
    "element",
    [
        None,
        "",
        "   ",
    ],
)
def test_missing_element_rejected(
    element,
):
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element=element,
    )

    with pytest.raises(
        GroundingError,
        match=(
            "requires element "
            "or position"
        ),
    ):
        bridge.ground(
            action,
            ocr_results=[],
            image_size=(
                2560,
                1440,
            ),
            region=(
                0,
                0,
                2560,
                1440,
            ),
        )


def test_target_not_found_has_specific_error():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    with pytest.raises(
        GroundingTargetNotFoundError,
        match="target not found",
    ) as exc_info:
        bridge.ground(
            action,
            ocr_results=[
                make_result(
                    text="Chrome"
                )
            ],
            image_size=(
                2560,
                1440,
            ),
            region=(
                0,
                0,
                2560,
                1440,
            ),
        )

    assert (
        exc_info.value.__cause__
        is not None
    )


def test_ambiguous_target_has_specific_error():
    bridge = GroundingBridge()

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    with pytest.raises(
        GroundingAmbiguousTargetError,
        match="ambiguous target",
    ) as exc_info:
        bridge.ground(
            action,
            ocr_results=[
                make_result(),
                make_result(),
            ],
            image_size=(
                2560,
                1440,
            ),
            region=(
                0,
                0,
                2560,
                1440,
            ),
        )

    assert (
        exc_info.value.__cause__
        is not None
    )


class ExplodingLocator:
    def find_one(
        self,
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            "locator failed"
        )


def test_unexpected_locator_failure_remains_generic_grounding_error():
    bridge = GroundingBridge(
        locator=ExplodingLocator()
    )

    action = GUIAction(
        action_type=ActionType.CLICK,
        element="Firefox",
    )

    with pytest.raises(
        GroundingError
    ) as exc_info:
        bridge.ground(
            action,
            ocr_results=[
                make_result()
            ],
            image_size=(
                2560,
                1440,
            ),
            region=(
                0,
                0,
                2560,
                1440,
            ),
        )

    assert not isinstance(
        exc_info.value,
        GroundingTargetNotFoundError,
    )

    assert not isinstance(
        exc_info.value,
        GroundingAmbiguousTargetError,
    )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )