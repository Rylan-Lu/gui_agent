from types import SimpleNamespace

import numpy as np
import pytest

import gui_agent.agent.action_transaction as transaction_module
import gui_agent.agent.executor as executor_module

from gui_agent.agent.action_adapter import (
    ActionValidationError,
)
from gui_agent.agent.action_runner import ActionRunner
from gui_agent.agent.action_transaction import (
    ActionTransaction,
)
from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.grounding import (
    GroundingAmbiguousTargetError,
    GroundingError,
    GroundingTargetNotFoundError,
)
from gui_agent.agent.observation import Observation
from gui_agent.agent.retry import RetryPolicy
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


class FakeEnvironment:
    def __init__(
        self,
        *,
        ocr_batches=None,
    ):
        self.calls = []
        self.index = 0

        self.ocr_batches = list(
            ocr_batches or []
        )

        self.ocr_index = 0

    def observe(
        self,
        *,
        use_ocr=False,
    ):
        self.calls.append(
            use_ocr
        )

        image = np.zeros(
            (
                100,
                100,
                3,
            ),
            dtype=np.uint8,
        )

        if self.index > 0:
            value = min(
                self.index * 50,
                255,
            )

            image[
                10:20,
                10:20,
            ] = value

        self.index += 1

        ocr_result = None

        if use_ocr:
            if self.ocr_batches:
                batch_index = min(
                    self.ocr_index,
                    len(
                        self.ocr_batches
                    ) - 1,
                )

                ocr_result = (
                    self.ocr_batches[
                        batch_index
                    ]
                )

                self.ocr_index += 1

            else:
                ocr_result = [
                    make_ocr_result(
                        "Firefox"
                    )
                ]

        return Observation(
            screenshot=image,
            screen_width=100,
            screen_height=100,
            ocr_result=ocr_result,
            metadata={
                "image_size": (
                    100,
                    100,
                ),
                "region": (
                    0,
                    0,
                    100,
                    100,
                ),
            },
        )


def make_ocr_result(
    text="Firefox",
    *,
    bbox=(
        (10, 10),
        (30, 10),
        (30, 30),
        (10, 30),
    ),
):
    return OCRResult(
        text=text,
        confidence=0.98,
        bbox=bbox,
    )


def make_transaction(
    *,
    settle_seconds=0,
    retry_policy=None,
    environment=None,
    runner=None,
):
    controller = FakeController()

    environment = (
        environment
        or FakeEnvironment()
    )

    runner = (
        runner
        or ActionRunner(
            ActionExecutor(
                controller
            )
        )
    )

    transaction = ActionTransaction(
        environment,
        runner,
        settle_seconds=(
            settle_seconds
        ),
        retry_policy=retry_policy,
    )

    return (
        controller,
        environment,
        transaction,
    )


def test_click_transaction():
    (
        controller,
        environment,
        transaction,
    ) = make_transaction()

    result = transaction.execute(
        GUIAction(
            action_type=(
                ActionType.CLICK
            ),
            element="Firefox",
        )
    )

    assert environment.calls == [
        True,
        False,
    ]

    assert controller.calls == [
        (
            "click",
            (
                20,
                20,
            ),
            "left",
        )
    ]

    assert (
        result.feedback.changed
        is True
    )

    assert (
        result.action.position
        == Point(
            20,
            20,
        )
    )

    assert (
        result.grounding_attempts
        == 1
    )

    assert (
        result.grounding_retries
        == 0
    )


def test_positioned_click_transaction_skips_ocr():
    (
        controller,
        environment,
        transaction,
    ) = make_transaction()

    result = transaction.execute(
        GUIAction(
            action_type=(
                ActionType.CLICK
            ),
            position=Point(
                25,
                35,
            ),
        )
    )

    assert environment.calls == [
        False,
        False,
    ]

    assert controller.calls == [
        (
            "click",
            (
                25,
                35,
            ),
            "left",
        )
    ]

    assert (
        result.action.position
        == Point(
            25,
            35,
        )
    )

    assert (
        result.grounding_attempts
        == 0
    )

    assert (
        result.grounding_retries
        == 0
    )


def test_type_text_transaction_without_ocr():
    (
        controller,
        environment,
        transaction,
    ) = make_transaction()

    result = transaction.execute(
        GUIAction(
            action_type=(
                ActionType.TYPE_TEXT
            ),
            text="hello",
        )
    )

    assert environment.calls == [
        False,
        False,
    ]

    assert controller.calls == [
        (
            "type_text",
            "hello",
        )
    ]

    assert (
        result.feedback.changed
        is True
    )

    assert (
        result.grounding_attempts
        == 0
    )


def test_invalid_click_fails_before_observation():
    (
        controller,
        environment,
        transaction,
    ) = make_transaction()

    with pytest.raises(
        ActionValidationError
    ):
        transaction.execute(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                )
            )
        )

    assert (
        environment.calls
        == []
    )

    assert (
        controller.calls
        == []
    )


def test_invalid_type_text_fails_before_observation():
    (
        controller,
        environment,
        transaction,
    ) = make_transaction()

    with pytest.raises(
        ActionValidationError
    ):
        transaction.execute(
            GUIAction(
                action_type=(
                    ActionType.TYPE_TEXT
                ),
                text=None,
            )
        )

    assert (
        environment.calls
        == []
    )

    assert (
        controller.calls
        == []
    )


def test_target_not_found_can_retry_with_fresh_observation(
    monkeypatch,
):
    environment = FakeEnvironment(
        ocr_batches=[
            [
                make_ocr_result(
                    "Chrome"
                )
            ],
            [
                make_ocr_result(
                    "Firefox"
                )
            ],
        ]
    )

    (
        controller,
        _,
        transaction,
    ) = make_transaction(
        environment=environment,
        retry_policy=RetryPolicy(
            max_grounding_attempts=2,
            grounding_retry_delay_s=0.2,
        ),
    )

    sleep_calls = []

    monkeypatch.setattr(
        transaction_module.time,
        "sleep",
        sleep_calls.append,
    )

    result = transaction.execute(
        GUIAction(
            action_type=(
                ActionType.CLICK
            ),
            element="Firefox",
        )
    )

    assert environment.calls == [
        True,
        True,
        False,
    ]

    assert (
        sleep_calls
        == [0.2]
    )

    # Only the successful grounded action
    # reaches the real Controller.
    assert controller.calls == [
        (
            "click",
            (
                20,
                20,
            ),
            "left",
        )
    ]

    assert (
        result.grounding_attempts
        == 2
    )

    assert (
        result.grounding_retries
        == 1
    )

    # Feedback must use the fresh observation
    # that actually produced the successful Grounding.
    assert (
        result.before
        .ocr_result[0]
        .text
        == "Firefox"
    )


def test_target_not_found_exhaustion_raises_without_desktop_action(
    monkeypatch,
):
    environment = FakeEnvironment(
        ocr_batches=[
            [
                make_ocr_result(
                    "Chrome"
                )
            ],
            [
                make_ocr_result(
                    "Edge"
                )
            ],
        ]
    )

    (
        controller,
        _,
        transaction,
    ) = make_transaction(
        environment=environment,
        retry_policy=RetryPolicy(
            max_grounding_attempts=2,
            grounding_retry_delay_s=0.1,
        ),
    )

    sleep_calls = []

    monkeypatch.setattr(
        transaction_module.time,
        "sleep",
        sleep_calls.append,
    )

    with pytest.raises(
        GroundingTargetNotFoundError
    ):
        transaction.execute(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
            )
        )

    assert environment.calls == [
        True,
        True,
    ]

    assert (
        sleep_calls
        == [0.1]
    )

    # Critical safety contract:
    # no Controller side effect occurred.
    assert (
        controller.calls
        == []
    )


def test_ambiguous_target_is_not_retried(
    monkeypatch,
):
    environment = FakeEnvironment(
        ocr_batches=[
            [
                make_ocr_result(
                    "Firefox"
                ),
                make_ocr_result(
                    "Firefox"
                ),
            ]
        ]
    )

    (
        controller,
        _,
        transaction,
    ) = make_transaction(
        environment=environment,
        retry_policy=RetryPolicy(
            max_grounding_attempts=3,
            grounding_retry_delay_s=0.2,
        ),
    )

    sleep_calls = []

    monkeypatch.setattr(
        transaction_module.time,
        "sleep",
        sleep_calls.append,
    )

    with pytest.raises(
        GroundingAmbiguousTargetError
    ):
        transaction.execute(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
            )
        )

    assert environment.calls == [
        True
    ]

    assert (
        sleep_calls
        == []
    )

    assert (
        controller.calls
        == []
    )


class AlwaysGenericGroundingFailure:
    def ground(
        self,
        *args,
        **kwargs,
    ):
        raise GroundingError(
            "generic grounding failure"
        )


def test_generic_grounding_error_is_not_retried(
    monkeypatch,
):
    environment = FakeEnvironment()

    controller = FakeController()

    runner = ActionRunner(
        ActionExecutor(
            controller
        ),
        grounding=(
            AlwaysGenericGroundingFailure()
        ),
    )

    transaction = ActionTransaction(
        environment,
        runner,
        settle_seconds=0,
        retry_policy=RetryPolicy(
            max_grounding_attempts=3,
            grounding_retry_delay_s=0.2,
        ),
    )

    sleep_calls = []

    monkeypatch.setattr(
        transaction_module.time,
        "sleep",
        sleep_calls.append,
    )

    with pytest.raises(
        GroundingError
    ):
        transaction.execute(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
            )
        )

    assert environment.calls == [
        True
    ]

    assert (
        sleep_calls
        == []
    )

    assert (
        controller.calls
        == []
    )


def test_default_retry_policy_preserves_one_attempt_behavior():
    environment = FakeEnvironment(
        ocr_batches=[
            [
                make_ocr_result(
                    "Chrome"
                )
            ]
        ]
    )

    (
        controller,
        _,
        transaction,
    ) = make_transaction(
        environment=environment
    )

    with pytest.raises(
        GroundingTargetNotFoundError
    ):
        transaction.execute(
            GUIAction(
                action_type=(
                    ActionType.CLICK
                ),
                element="Firefox",
            )
        )

    assert environment.calls == [
        True
    ]

    assert (
        controller.calls
        == []
    )


def test_normal_action_applies_settle_delay(
    monkeypatch,
):
    (
        _,
        _,
        transaction,
    ) = make_transaction(
        settle_seconds=0.25
    )

    sleep_calls = []

    monkeypatch.setattr(
        transaction_module,
        "time",
        SimpleNamespace(
            sleep=(
                sleep_calls.append
            )
        ),
    )

    transaction.execute(
        GUIAction(
            action_type=(
                ActionType.TYPE_TEXT
            ),
            text="hello",
        )
    )

    assert (
        sleep_calls
        == [0.25]
    )


def test_wait_does_not_add_transaction_settle_delay(
    monkeypatch,
):
    (
        _,
        environment,
        transaction,
    ) = make_transaction(
        settle_seconds=0.5
    )

    executor_sleep_calls = []
    transaction_sleep_calls = []

    monkeypatch.setattr(
        executor_module,
        "time",
        SimpleNamespace(
            sleep=(
                executor_sleep_calls.append
            )
        ),
    )

    monkeypatch.setattr(
        transaction_module,
        "time",
        SimpleNamespace(
            sleep=(
                transaction_sleep_calls.append
            )
        ),
    )

    transaction.execute(
        GUIAction(
            action_type=(
                ActionType.WAIT
            ),
            wait_seconds=1.25,
        )
    )

    assert (
        executor_sleep_calls
        == [1.25]
    )

    assert (
        transaction_sleep_calls
        == []
    )

    assert environment.calls == [
        False,
        False,
    ]


@pytest.mark.parametrize(
    "settle_seconds",
    [
        -0.01,
        True,
        "0.3",
        float("inf"),
        float("nan"),
    ],
)
def test_invalid_settle_seconds_rejected(
    settle_seconds,
):
    controller = FakeController()
    environment = FakeEnvironment()

    with pytest.raises(
        ValueError,
        match="settle_seconds",
    ):
        ActionTransaction(
            environment,
            ActionRunner(
                ActionExecutor(
                    controller
                )
            ),
            settle_seconds=(
                settle_seconds
            ),
        )


def test_integer_settle_seconds_is_accepted():
    (
        _,
        _,
        transaction,
    ) = make_transaction(
        settle_seconds=1
    )

    assert (
        transaction.settle_seconds
        == 1.0
    )


def test_invalid_retry_policy_type_rejected():
    controller = FakeController()
    environment = FakeEnvironment()

    with pytest.raises(
        TypeError,
        match="retry_policy",
    ):
        ActionTransaction(
            environment,
            ActionRunner(
                ActionExecutor(
                    controller
                )
            ),
            retry_policy="retry",
        )