import math

import pytest

from gui_agent.agent.retry import RetryPolicy


def test_retry_policy_defaults_preserve_single_attempt_behavior():
    policy = RetryPolicy()

    assert (
        policy.max_grounding_attempts
        == 1
    )

    assert (
        policy.grounding_retry_delay_s
        == 0.2
    )


def test_retry_policy_accepts_multiple_attempts():
    policy = RetryPolicy(
        max_grounding_attempts=2,
        grounding_retry_delay_s=0,
    )

    assert (
        policy.max_grounding_attempts
        == 2
    )

    assert (
        policy.grounding_retry_delay_s
        == 0.0
    )


@pytest.mark.parametrize(
    "max_grounding_attempts",
    [
        0,
        -1,
        True,
        1.5,
        "2",
        None,
    ],
)
def test_invalid_max_grounding_attempts_rejected(
    max_grounding_attempts,
):
    with pytest.raises(
        ValueError,
        match=(
            "max_grounding_attempts"
        ),
    ):
        RetryPolicy(
            max_grounding_attempts=(
                max_grounding_attempts
            )
        )


@pytest.mark.parametrize(
    "grounding_retry_delay_s",
    [
        -0.01,
        True,
        "0.2",
        None,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_grounding_retry_delay_rejected(
    grounding_retry_delay_s,
):
    with pytest.raises(
        ValueError,
        match=(
            "grounding_retry_delay_s"
        ),
    ):
        RetryPolicy(
            grounding_retry_delay_s=(
                grounding_retry_delay_s
            )
        )


def test_integer_retry_delay_is_normalized_to_float():
    policy = RetryPolicy(
        grounding_retry_delay_s=1
    )

    assert (
        policy.grounding_retry_delay_s
        == 1.0
    )

    assert isinstance(
        policy.grounding_retry_delay_s,
        float,
    )