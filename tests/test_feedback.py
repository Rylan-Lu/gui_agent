import numpy as np
import pytest

from gui_agent.agent.feedback import (
    ActionFeedback,
    FeedbackError,
)
from gui_agent.agent.observation import Observation


def make_observation(image):
    height, width = image.shape[:2]

    return Observation(
        screenshot=image,
        screen_width=width,
        screen_height=height,
    )


def test_no_change():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    feedback = ActionFeedback()

    result = feedback.compare(
        make_observation(image.copy()),
        make_observation(image.copy()),
    )

    assert result.changed is False
    assert result.changed_pixel_ratio == 0.0
    assert result.mean_abs_diff == 0.0


def test_detect_change():
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()

    after[10:20, 10:20] = 255

    feedback = ActionFeedback()

    result = feedback.compare(
        make_observation(before),
        make_observation(after),
    )

    assert result.changed is True
    assert result.changed_pixel_ratio == pytest.approx(0.01)
    assert result.mean_abs_diff > 0


def test_ignore_tiny_pixel_difference():
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()

    after[10:20, 10:20] = 3

    feedback = ActionFeedback(
        pixel_threshold=8,
    )

    result = feedback.compare(
        make_observation(before),
        make_observation(after),
    )

    assert result.changed is False


def test_ratio_threshold():
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()

    after[0, 0] = 255

    feedback = ActionFeedback(
        changed_ratio_threshold=0.01,
    )

    result = feedback.compare(
        make_observation(before),
        make_observation(after),
    )

    assert result.changed is False


def test_different_shapes_rejected():
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = np.zeros((200, 100, 3), dtype=np.uint8)

    feedback = ActionFeedback()

    with pytest.raises(FeedbackError):
        feedback.compare(
            make_observation(before),
            make_observation(after),
        )


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        ActionFeedback(pixel_threshold=-1)

    with pytest.raises(ValueError):
        ActionFeedback(changed_ratio_threshold=2)