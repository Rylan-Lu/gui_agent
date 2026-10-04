from __future__ import annotations

import math

from gui_agent.agent.plan_schema import PlanStep
from gui_agent.datasets.schema import (
    ActionType,
    CoordinateSpace,
    GUIAction,
    Point,
)


class ActionValidationError(
    ValueError
):
    pass


EXECUTABLE_ACTION_TYPES = frozenset(
    {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.MOVE,
        ActionType.DRAG,
        ActionType.SCROLL,
        ActionType.TYPE_TEXT,
        ActionType.KEY_PRESS,
        ActionType.HOTKEY,
        ActionType.WAIT,
    }
)


POSITIONAL_ACTION_TYPES = frozenset(
    {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.MOVE,
        ActionType.DRAG,
    }
)


_SUPPORTED_MOUSE_BUTTONS = {
    None,
    "left",
    "right",
    "middle",
}


def plan_step_to_action(
    step: PlanStep,
) -> GUIAction:
    if step.action is None:
        raise ActionValidationError(
            f"plan step {step.step_id} "
            "has no action"
        )

    return step.action


def is_executable_action(
    action: GUIAction,
) -> bool:
    return (
        action.action_type
        in EXECUTABLE_ACTION_TYPES
    )


def _validate_point(
    point: Point | None,
    *,
    field_name: str,
) -> None:
    if not isinstance(
        point,
        Point,
    ):
        raise ActionValidationError(
            f"{field_name} must be a Point"
        )

    for (
        coordinate_name,
        value,
    ) in (
        ("x", point.x),
        ("y", point.y),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise ActionValidationError(
                f"{field_name}."
                f"{coordinate_name} "
                "must be a number"
            )

        if not math.isfinite(
            float(value)
        ):
            raise ActionValidationError(
                f"{field_name}."
                f"{coordinate_name} "
                "must be finite"
            )


def _validate_keys(
    keys: tuple[str, ...],
    *,
    exact_count: int | None = None,
) -> None:
    if not isinstance(
        keys,
        tuple,
    ):
        raise ActionValidationError(
            "keys must be a tuple"
        )

    if not keys:
        raise ActionValidationError(
            "keys must not be empty"
        )

    if not all(
        isinstance(key, str)
        and key.strip()
        for key in keys
    ):
        raise ActionValidationError(
            "keys must contain "
            "non-empty strings"
        )

    if (
        exact_count is not None
        and len(keys) != exact_count
    ):
        raise ActionValidationError(
            "keys must contain exactly "
            f"{exact_count} item(s)"
        )


def validate_executable_action(
    action: GUIAction,
) -> GUIAction:
    """
    Validate an action immediately before
    it reaches ActionExecutor.

    This is the runtime execution contract,
    not the broad dataset GUIAction schema.
    """

    if not isinstance(
        action,
        GUIAction,
    ):
        raise ActionValidationError(
            "action must be a GUIAction"
        )

    if not is_executable_action(
        action
    ):
        raise ActionValidationError(
            "action type is not directly "
            "executable: "
            f"{action.action_type.value}"
        )

    action_type = (
        action.action_type
    )

    # Only actions that actually carry
    # positions care about coordinate space.
    if (
        action_type
        in POSITIONAL_ACTION_TYPES
        and action.coordinate_space
        != CoordinateSpace.PIXEL
    ):
        raise ActionValidationError(
            f"{action_type.value} requires "
            "pixel coordinates before execution"
        )

    if action_type in {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.MOVE,
    }:
        _validate_point(
            action.position,
            field_name="position",
        )

    elif (
        action_type
        == ActionType.DRAG
    ):
        _validate_point(
            action.position,
            field_name="position",
        )

        _validate_point(
            action.end_position,
            field_name="end_position",
        )

    elif (
        action_type
        == ActionType.SCROLL
    ):
        if (
            type(
                action.scroll_delta
            )
            is not int
        ):
            raise ActionValidationError(
                "scroll_delta must be an integer"
            )

    elif (
        action_type
        == ActionType.TYPE_TEXT
    ):
        if not isinstance(
            action.text,
            str,
        ):
            raise ActionValidationError(
                "type_text requires string text"
            )

    elif (
        action_type
        == ActionType.KEY_PRESS
    ):
        _validate_keys(
            action.keys,
            exact_count=1,
        )

    elif (
        action_type
        == ActionType.HOTKEY
    ):
        _validate_keys(
            action.keys
        )

    elif (
        action_type
        == ActionType.WAIT
    ):
        wait_seconds = (
            action.wait_seconds
        )

        if (
            isinstance(
                wait_seconds,
                bool,
            )
            or not isinstance(
                wait_seconds,
                (int, float),
            )
        ):
            raise ActionValidationError(
                "wait_seconds must be a number"
            )

        wait_seconds = float(
            wait_seconds
        )

        if not math.isfinite(
            wait_seconds
        ):
            raise ActionValidationError(
                "wait_seconds must be finite"
            )

        if wait_seconds < 0:
            raise ActionValidationError(
                "wait_seconds must be "
                "non-negative"
            )

    if (
        action.mouse_button
        not in _SUPPORTED_MOUSE_BUTTONS
    ):
        raise ActionValidationError(
            "unsupported mouse button: "
            f"{action.mouse_button}"
        )

    return action