from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True,
)
class RetryPolicy:
    """
    Retry policy for safe pre-execution grounding retries.

    max_grounding_attempts includes the first attempt.

    Examples:
        1 -> no retry
        2 -> one retry
        3 -> at most two retries
    """

    max_grounding_attempts: int = 1
    grounding_retry_delay_s: float = 0.2

    def __post_init__(self) -> None:
        if (
            type(self.max_grounding_attempts) is not int
            or self.max_grounding_attempts <= 0
        ):
            raise ValueError(
                "max_grounding_attempts "
                "must be a positive integer"
            )

        delay = self.grounding_retry_delay_s

        if (
            isinstance(delay, bool)
            or not isinstance(
                delay,
                (int, float),
            )
        ):
            raise ValueError(
                "grounding_retry_delay_s "
                "must be a number"
            )

        delay = float(delay)

        if not math.isfinite(delay):
            raise ValueError(
                "grounding_retry_delay_s "
                "must be finite"
            )

        if delay < 0:
            raise ValueError(
                "grounding_retry_delay_s "
                "must be non-negative"
            )

        object.__setattr__(
            self,
            "grounding_retry_delay_s",
            delay,
        )