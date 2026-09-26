from __future__ import annotations

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
        if pixel_threshold < 0:
            raise ValueError("pixel_threshold must be non-negative")

        if not 0 <= changed_ratio_threshold <= 1:
            raise ValueError(
                "changed_ratio_threshold must be between 0 and 1"
            )

        self.pixel_threshold = pixel_threshold
        self.changed_ratio_threshold = changed_ratio_threshold

    def compare(
        self,
        before: Observation,
        after: Observation,
    ) -> FeedbackResult:

        before_image = np.asarray(before.screenshot)
        after_image = np.asarray(after.screenshot)

        if before_image.shape != after_image.shape:
            raise FeedbackError(
                "before and after screenshots must have the same shape"
            )

        before_image = before_image.astype(np.int16)
        after_image = after_image.astype(np.int16)

        diff = np.abs(after_image - before_image)

        if diff.ndim == 3:
            pixel_diff = diff.max(axis=2)
        else:
            pixel_diff = diff

        changed_mask = pixel_diff >= self.pixel_threshold

        changed_pixel_ratio = float(
            np.count_nonzero(changed_mask)
            / changed_mask.size
        )

        mean_abs_diff = float(diff.mean())

        return FeedbackResult(
            changed=(
                changed_pixel_ratio
                >= self.changed_ratio_threshold
            ),
            changed_pixel_ratio=changed_pixel_ratio,
            mean_abs_diff=mean_abs_diff,
        )