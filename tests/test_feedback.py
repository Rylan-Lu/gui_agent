import numpy as np
import pytest

from gui_agent.agent.feedback import (
    ActionFeedback,
    FeedbackError,
)
from gui_agent.agent.observation import Observation


def make_observation(
    image,
):
    height, width = (
        image.shape[:2]
    )

    return Observation(
        screenshot=image,
        screen_width=width,
        screen_height=height,
    )


def test_no_change():
    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    feedback = ActionFeedback()

    result = feedback.compare(
        make_observation(
            image.copy()
        ),
        make_observation(
            image.copy()
        ),
    )

    assert (
        result.changed
        is False
    )

    assert (
        result.changed_pixel_ratio
        == 0.0
    )

    assert (
        result.mean_abs_diff
        == 0.0
    )


def test_detect_change():
    before = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    after = before.copy()

    after[
        10:20,
        10:20,
    ] = 255

    feedback = ActionFeedback()

    result = feedback.compare(
        make_observation(
            before
        ),
        make_observation(
            after
        ),
    )

    assert (
        result.changed
        is True
    )

    assert (
        result.changed_pixel_ratio
        == pytest.approx(
            0.01
        )
    )

    assert (
        result.mean_abs_diff
        > 0
    )


def test_ignore_tiny_pixel_difference():
    before = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    after = before.copy()

    after[
        10:20,
        10:20,
    ] = 3

    feedback = ActionFeedback(
        pixel_threshold=8,
    )

    result = feedback.compare(
        make_observation(
            before
        ),
        make_observation(
            after
        ),
    )

    assert (
        result.changed
        is False
    )

    assert (
        result.changed_pixel_ratio
        == 0.0
    )


def test_pixel_difference_equal_to_threshold_counts_as_change():
    before = np.zeros(
        (
            10,
            10,
            3,
        ),
        dtype=np.uint8,
    )

    after = before.copy()

    after[
        0,
        0,
    ] = 8

    feedback = ActionFeedback(
        pixel_threshold=8,
        changed_ratio_threshold=0.01,
    )

    result = feedback.compare(
        make_observation(
            before
        ),
        make_observation(
            after
        ),
    )

    assert (
        result.changed_pixel_ratio
        == pytest.approx(
            0.01
        )
    )

    assert (
        result.changed
        is True
    )


def test_ratio_threshold():
    before = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    after = before.copy()

    after[
        0,
        0,
    ] = 255

    feedback = ActionFeedback(
        changed_ratio_threshold=0.01,
    )

    result = feedback.compare(
        make_observation(
            before
        ),
        make_observation(
            after
        ),
    )

    assert (
        result.changed
        is False
    )


def test_ratio_equal_to_threshold_counts_as_change():
    before = np.zeros(
        (
            10,
            10,
            3,
        ),
        dtype=np.uint8,
    )

    after = before.copy()

    after[
        0,
        0,
    ] = 255

    feedback = ActionFeedback(
        changed_ratio_threshold=0.01,
    )

    result = feedback.compare(
        make_observation(
            before
        ),
        make_observation(
            after
        ),
    )

    assert (
        result.changed_pixel_ratio
        == pytest.approx(
            0.01
        )
    )

    assert (
        result.changed
        is True
    )


def test_grayscale_images_are_supported():
    before = np.zeros(
        (
            10,
            10,
        ),
        dtype=np.uint8,
    )

    after = before.copy()

    after[
        0,
        0,
    ] = 255

    feedback = ActionFeedback(
        changed_ratio_threshold=0.01,
    )

    result = feedback.compare(
        make_observation(
            before
        ),
        make_observation(
            after
        ),
    )

    assert (
        result.changed
        is True
    )

    assert (
        result.changed_pixel_ratio
        == pytest.approx(
            0.01
        )
    )


def test_different_shapes_rejected():
    before = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    after = np.zeros(
        (
            200,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    feedback = ActionFeedback()

    with pytest.raises(
        FeedbackError,
        match="same shape",
    ):
        feedback.compare(
            make_observation(
                before
            ),
            make_observation(
                after
            ),
        )


@pytest.mark.parametrize(
    "pixel_threshold",
    [
        -1,
        0,
        256,
        1.5,
        "8",
        True,
        None,
    ],
)
def test_invalid_pixel_thresholds_rejected(
    pixel_threshold,
):
    with pytest.raises(
        ValueError,
        match="pixel_threshold",
    ):
        ActionFeedback(
            pixel_threshold=(
                pixel_threshold
            ),
        )


@pytest.mark.parametrize(
    "changed_ratio_threshold",
    [
        -0.1,
        0,
        1.01,
        "0.1",
        True,
        None,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_invalid_changed_ratio_thresholds_rejected(
    changed_ratio_threshold,
):
    with pytest.raises(
        ValueError,
        match=(
            "changed_ratio_threshold"
        ),
    ):
        ActionFeedback(
            changed_ratio_threshold=(
                changed_ratio_threshold
            ),
        )


@pytest.mark.parametrize(
    "pixel_threshold",
    [
        1,
        8,
        255,
    ],
)
def test_valid_pixel_threshold_boundaries(
    pixel_threshold,
):
    feedback = ActionFeedback(
        pixel_threshold=(
            pixel_threshold
        ),
    )

    assert (
        feedback.pixel_threshold
        == pixel_threshold
    )


@pytest.mark.parametrize(
    "changed_ratio_threshold",
    [
        0.0001,
        0.5,
        1,
        1.0,
    ],
)
def test_valid_changed_ratio_threshold_boundaries(
    changed_ratio_threshold,
):
    feedback = ActionFeedback(
        changed_ratio_threshold=(
            changed_ratio_threshold
        ),
    )

    assert (
        feedback.changed_ratio_threshold
        == float(
            changed_ratio_threshold
        )
    )