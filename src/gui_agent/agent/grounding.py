from __future__ import annotations

from dataclasses import replace

from gui_agent.datasets.schema import (
    ActionType,
    CoordinateSpace,
    GUIAction,
    Point,
)
from gui_agent.locator.coordinate_mapper import CoordinateMapper
from gui_agent.locator.ui_locator import (
    AmbiguousTargetError,
    MatchMode,
    TargetNotFoundError,
    UILocator,
)
from gui_agent.ocr.base import OCRResult


class GroundingError(RuntimeError):
    pass


class GroundingTargetNotFoundError(GroundingError):
    pass


class GroundingAmbiguousTargetError(GroundingError):
    pass


GROUNDABLE_ACTION_TYPES = {
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
}


class GroundingBridge:
    def __init__(
        self,
        locator: UILocator | None = None,
    ):
        self.locator = locator or UILocator()

    def ground(
        self,
        action: GUIAction,
        *,
        ocr_results: list[OCRResult],
        image_size: tuple[int, int],
        region: tuple[int, int, int, int],
        mode: MatchMode = "exact",
        min_confidence: float = 0.0,
        fuzzy_threshold: float = 0.8,
    ) -> GUIAction:
        if action.action_type not in GROUNDABLE_ACTION_TYPES:
            return action

        if action.position is not None:
            return action

        if (
            not isinstance(action.element, str)
            or not action.element.strip()
        ):
            raise GroundingError(
                f"{action.action_type.value} "
                "requires element or position"
            )

        try:
            located = self.locator.find_one(
                ocr_results,
                action.element,
                mode=mode,
                min_confidence=min_confidence,
                fuzzy_threshold=fuzzy_threshold,
            )

        except TargetNotFoundError as exc:
            raise GroundingTargetNotFoundError(
                "target not found while grounding "
                f"element {action.element!r}: {exc}"
            ) from exc

        except AmbiguousTargetError as exc:
            raise GroundingAmbiguousTargetError(
                "ambiguous target while grounding "
                f"element {action.element!r}: {exc}"
            ) from exc

        except Exception as exc:
            raise GroundingError(
                "failed to locate element "
                f"{action.element!r}: {exc}"
            ) from exc

        try:
            desktop_x, desktop_y = (
                CoordinateMapper.image_to_desktop(
                    located.image_center,
                    image_size=image_size,
                    region=region,
                )
            )

        except Exception as exc:
            raise GroundingError(
                "failed to map element "
                f"{action.element!r} "
                "to desktop coordinates: "
                f"{exc}"
            ) from exc

        return replace(
            action,
            position=Point(
                desktop_x,
                desktop_y,
            ),
            coordinate_space=CoordinateSpace.PIXEL,
        )