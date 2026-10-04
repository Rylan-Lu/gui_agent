from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gui_agent.agent.observation import Observation


class FeedbackError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeedbackResult:
    changed: bool
    changed_pixel_ratio: float
    mean_abs_diff: float


class ActionFeedback:
    def __init__(
        self,
        *,
        pixel_threshold: int = 8,
        changed_ratio_threshold: float = 0.0001,
    ):
        """
        Compare two screenshots using a simple pixel-difference signal.

        pixel_threshold:
            Minimum per-pixel channel difference required for a pixel
            to count as changed.

            Valid range: 1..255

        changed_ratio_threshold:
            Minimum fraction of changed pixels required for the entire
            screenshot to count as changed.

            Valid range: 0 < value <= 1
        """

        # bool is a subclass of int, so exact type checking is
        # intentional here.
        if type(pixel_threshold) is not int:
            raise ValueError(
                "pixel_threshold must be an integer"
            )

        if not 1 <= pixel_threshold <= 255:
            raise ValueError(
                "pixel_threshold must be between 1 and 255"
            )

        if (
            isinstance(
                changed_ratio_threshold,
                bool,
            )
            or not isinstance(
                changed_ratio_threshold,
                (int, float),
            )
        ):
            raise ValueError(
                "changed_ratio_threshold must be a number"
            )

        changed_ratio_threshold = float(
            changed_ratio_threshold
        )

        if not math.isfinite(
            changed_ratio_threshold
        ):
            raise ValueError(
                "changed_ratio_threshold must be finite"
            )

        if not (
            0
            < changed_ratio_threshold
            <= 1
        ):
            raise ValueError(
                "changed_ratio_threshold must be "
                "greater than 0 and at most 1"
            )

        self.pixel_threshold = pixel_threshold
        self.changed_ratio_threshold = (
            changed_ratio_threshold
        )

    def compare(
        self,
        before: Observation,
        after: Observation,
    ) -> FeedbackResult:
        before_image = np.asarray(
            before.screenshot
        )

        after_image = np.asarray(
            after.screenshot
        )

        if (
            before_image.shape
            != after_image.shape
        ):
            raise FeedbackError(
                "before and after screenshots "
                "must have the same shape"
            )

        # int16 prevents uint8 subtraction wraparound.
        before_image = before_image.astype(
            np.int16
        )

        after_image = after_image.astype(
            np.int16
        )

        diff = np.abs(
            after_image
            - before_image
        )

        if diff.ndim == 3:
            # A pixel is considered changed if any channel differs
            # by at least pixel_threshold.
            pixel_diff = diff.max(
                axis=2
            )

        else:
            pixel_diff = diff

        changed_mask = (
            pixel_diff
            >= self.pixel_threshold
        )

        changed_pixel_ratio = float(
            np.count_nonzero(
                changed_mask
            )
            / changed_mask.size
        )

        mean_abs_diff = float(
            diff.mean()
        )

        return FeedbackResult(
            changed=(
                changed_pixel_ratio
                >= self.changed_ratio_threshold
            ),
            changed_pixel_ratio=(
                changed_pixel_ratio
            ),
            mean_abs_diff=(
                mean_abs_diff
            ),
        )