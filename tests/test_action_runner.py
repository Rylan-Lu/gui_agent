import numpy as np
import pytest

from gui_agent.agent.action_adapter import (
    ActionValidationError,
)
from gui_agent.agent.action_runner import (
    ActionRunRequirements,
    ActionRunner,
    ActionRunnerError,
)
from gui_agent.agent.executor import (
    ActionExecutionError,
    ActionExecutor,
)
from gui_agent.agent.grounding import (
    GroundingError,
)
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


def make_observation(
    *,
    ocr_result="default",
    metadata=None,
):
    if ocr_result == "default":
        ocr_result = [
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
        ]

    if metadata is None:
        metadata = {
            "image_size": (
                2560,
                1440,
            ),
            "region": (
                0,
                0,
                2560,
                1440,
            ),
        }

    return Observation(
        screenshot=np.zeros(
            (
                1440,
                2560,
                3,
            ),
            dtype=np.uint8,
        ),
        screen_width=2560,
        screen_height=1440,
        ocr_result=ocr_result,
        metadata=metadata,
    )


def make_runner(
    controller=None,
):
    controller = (
        controller
        or FakeController()
    )

    runner = ActionRunner(
        ActionExecutor(
            controller
        )
    )

    return (
        controller,
        runner,
    )


def test_preflight_unresolved_click_requires_ocr():
    _, runner = make_runner()

    requirements = (
        runner.preflight(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
            )
        )
    )

    assert requirements == (
        ActionRunRequirements(
            needs_ocr=True
        )
    )


def test_preflight_positioned_click_skips_ocr():
    _, runner = make_runner()

    requirements = (
        runner.preflight(
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
    )

    assert (
        requirements.needs_ocr
        is False
    )


def test_preflight_coordinate_free_action_skips_ocr():
    _, runner = make_runner()

    requirements = (
        runner.preflight(
            GUIAction(
                action_type=(
                    ActionType.TYPE_TEXT
                ),
                text="hello",
            )
        )
    )

    assert (
        requirements.needs_ocr
        is False
    )


@pytest.mark.parametrize(
    "element",
    [
        None,
        "",
        "   ",
    ],
)
def test_preflight_rejects_unresolvable_click(
    element,
):
    _, runner = make_runner()

    with pytest.raises(
        ActionValidationError,
        match="non-empty element",
    ):
        runner.preflight(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element=element,
            )
        )


def test_preflight_rejects_invalid_unresolved_mouse_button():
    _, runner = make_runner()

    with pytest.raises(
        ActionValidationError,
        match="mouse button",
    ):
        runner.preflight(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
                mouse_button="invalid",
            )
        )


def test_preflight_rejects_invalid_non_grounded_action():
    _, runner = make_runner()

    with pytest.raises(
        ActionValidationError
    ):
        runner.preflight(
            GUIAction(
                action_type=(
                    ActionType.TYPE_TEXT
                ),
                text=None,
            )
        )


def test_ground_and_execute_click():
    controller, runner = (
        make_runner()
    )

    action = GUIAction(
        action_type=(
            ActionType.CLICK
        ),
        element="Firefox",
    )

    resolved = runner.run(
        action,
        make_observation(),
    )

    assert (
        resolved.position
        == Point(
            150,
            120,
        )
    )

    assert controller.calls == [
        (
            "click",
            (
                150,
                120,
            ),
            "left",
        )
    ]


def test_ground_and_execute_double_click():
    controller, runner = (
        make_runner()
    )

    resolved = runner.run(
        GUIAction(
            action_type=(
                ActionType.DOUBLE_CLICK
            ),
            element="Firefox",
        ),
        make_observation(),
    )

    assert (
        resolved.position
        == Point(
            150,
            120,
        )
    )

    assert controller.calls == [
        (
            "double_click",
            (
                150,
                120,
            ),
            "left",
        )
    ]


def test_execute_action_without_grounding():
    controller, runner = (
        make_runner()
    )

    action = GUIAction(
        action_type=(
            ActionType.TYPE_TEXT
        ),
        text="hello",
    )

    resolved = runner.run(
        action,
        make_observation(),
    )

    assert resolved is action

    assert controller.calls == [
        (
            "type_text",
            "hello",
        )
    ]


def test_unresolved_click_requires_ocr_results():
    _, runner = make_runner()

    with pytest.raises(
        ActionRunnerError,
        match="OCR results",
    ):
        runner.run(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
            ),
            make_observation(
                ocr_result=None
            ),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "image_size": (
                2560,
                1440,
            ),
        },
        {
            "region": (
                0,
                0,
                2560,
                1440,
            ),
        },
    ],
)
def test_unresolved_click_requires_observation_metadata(
    metadata,
):
    _, runner = make_runner()

    with pytest.raises(
        ActionRunnerError,
        match=(
            "image_size or region"
        ),
    ):
        runner.run(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
            ),
            make_observation(
                metadata=metadata
            ),
        )


def test_existing_position_skips_grounding():
    controller, runner = (
        make_runner()
    )

    action = GUIAction(
        action_type=(
            ActionType.CLICK
        ),
        position=Point(
            500,
            600,
        ),
    )

    resolved = runner.run(
        action,
        make_observation(
            ocr_result=None,
            metadata={},
        ),
    )

    assert resolved is action

    assert controller.calls == [
        (
            "click",
            (
                500,
                600,
            ),
            "left",
        )
    ]


def test_grounding_error_is_preserved():
    _, runner = make_runner()

    with pytest.raises(
        GroundingError
    ):
        runner.run(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Missing",
            ),
            make_observation(),
        )


def test_execution_error_is_preserved():
    _, runner = make_runner(
        FailingController()
    )

    with pytest.raises(
        ActionExecutionError,
        match=(
            "failed to execute click"
        ),
    ):
        runner.run(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                position=Point(
                    100,
                    200,
                ),
            ),
            make_observation(
                ocr_result=None,
                metadata={},
            ),
        )